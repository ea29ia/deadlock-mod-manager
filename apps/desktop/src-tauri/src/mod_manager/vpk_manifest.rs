use crate::errors::Error;
use crate::mod_manager::profile_lock::{ProfileGuard, lock_profile};
use crate::mod_manager::shard::{
  CLEAR_STAGING_DIR, MANIFEST_TEMP_FILENAME, MANIFEST_TEMP_PREFIX, MANIFEST_TEMP_SUFFIX,
  ProfileBase, REORDER_STAGING_DIR, ShardIndex, UPDATE_STAGING_PREFIX,
};
use crate::mod_manager::vpk_manager::staging::{RecoveryMode, recover_staging_directory};
use serde::{Deserialize, Serialize};
use std::{
  collections::BTreeMap,
  fs,
  io::{ErrorKind, Write},
  ops::{Deref, DerefMut},
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
  time::{Duration, SystemTime, UNIX_EPOCH},
};

mod legacy_reorder;

const MANIFEST_FILENAME: &str = ".dmm.json";
const CURRENT_MANIFEST_VERSION: u32 = 3;

/// Distinguishes the temp files of saves running in this process.
static MANIFEST_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// How long a temp file written by another live process is left alone before it
/// is treated as abandoned. A commit holds the temp file for a single fsync.
const MANIFEST_TEMP_TTL: Duration = Duration::from_secs(60);

/// Path for the temp file one [`ProfileVpkManifest::save`] commits through.
///
/// The name carries the writing process and a monotonic counter so that two
/// saves never share a file, in this process or in another one.
fn next_manifest_temp_path(addons_path: &Path) -> PathBuf {
  let counter = MANIFEST_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|elapsed| elapsed.as_nanos())
    .unwrap_or(0);

  addons_path.join(format!(
    "{MANIFEST_FILENAME}.{}-{nanos}-{counter}.tmp",
    std::process::id()
  ))
}

/// Process that wrote a per-save temp file, if the name is one of ours.
fn temp_path_pid(name: &str) -> Option<u32> {
  let suffix = name.strip_prefix(MANIFEST_TEMP_PREFIX)?;
  let spec = suffix.strip_suffix(MANIFEST_TEMP_SUFFIX)?;
  let pid = spec.split('-').next()?;
  pid.parse().ok()
}

/// Whether a temp file left on disk can be deleted.
///
/// Deciding this from the name is what makes the cleanup safe across processes:
/// the profile write lock only serializes writers inside this process, so a
/// temp file belonging to a live peer may well be a commit in flight.
fn should_remove_temp_file(
  pid: Option<u32>,
  own_pid: u32,
  age: Duration,
  writer_is_alive: bool,
) -> bool {
  if pid == Some(own_pid) {
    // We hold the profile lock, so nothing of ours can be in flight any more.
    return true;
  }

  if pid.is_none() {
    // A name from an older or newer format: only reclaim it once it is
    // unambiguously old.
    return age > MANIFEST_TEMP_TTL;
  }

  !writer_is_alive || age > MANIFEST_TEMP_TTL
}

fn process_is_alive(pid: u32) -> bool {
  #[cfg(target_os = "linux")]
  {
    Path::new(&format!("/proc/{pid}")).exists()
  }

  // No portable way to probe another process; the age check covers it.
  #[cfg(not(target_os = "linux"))]
  {
    let _ = pid;
    true
  }
}

/// Manifest read through [`ProfileVpkManifest::open_for_write`], holding the
/// profile write lock for as long as it is alive.
///
/// Dereferences to the manifest so existing call sites keep working unchanged.
#[derive(Debug)]
pub struct ManifestWriteGuard {
  manifest: ProfileVpkManifest,
  _profile_lock: ProfileGuard,
}

impl Deref for ManifestWriteGuard {
  type Target = ProfileVpkManifest;

  fn deref(&self) -> &Self::Target {
    &self.manifest
  }
}

impl DerefMut for ManifestWriteGuard {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.manifest
  }
}

impl PartialEq<ProfileVpkManifest> for ManifestWriteGuard {
  fn eq(&self, other: &ProfileVpkManifest) -> bool {
    self.manifest == *other
  }
}

impl PartialEq<ManifestWriteGuard> for ProfileVpkManifest {
  fn eq(&self, other: &ManifestWriteGuard) -> bool {
    *self == other.manifest
  }
}

const fn current_manifest_version() -> u32 {
  CURRENT_MANIFEST_VERSION
}

/// Shard index a mod's enabled VPKs live in when the entry predates sharding
/// (manifest v1) or omits the field. Shard 1 is the profile base directory.
const fn default_shard() -> ShardIndex {
  ShardIndex::FIRST
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProfileVpkManifest {
  #[serde(default = "current_manifest_version")]
  pub version: u32,
  #[serde(default)]
  pub mods: BTreeMap<String, ProfileVpkManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProfileVpkManifestEntry {
  #[serde(default)]
  pub enabled: bool,
  #[serde(default)]
  pub compatibility_enabled: bool,
  /// The player chose this mod's original files over available fixes. Unlike
  /// `compatibility_enabled = false`, which is also the default, this records a
  /// decision that bulk fixes must respect.
  #[serde(default)]
  pub compatibility_opted_out: bool,
  #[serde(default)]
  pub order: Option<u32>,
  /// 1-based shard index the enabled VPKs of this mod currently live in.
  /// All VPKs of a mod always share one shard so multi-file mods stay together.
  #[serde(default = "default_shard")]
  pub shard: ShardIndex,
  #[serde(default)]
  pub current_vpks: Vec<String>,
  #[serde(default)]
  pub disabled_vpks: Vec<String>,
  #[serde(default)]
  pub original_vpk_names: Vec<String>,
}

impl Default for ProfileVpkManifestEntry {
  fn default() -> Self {
    Self {
      enabled: false,
      compatibility_enabled: false,
      compatibility_opted_out: false,
      order: None,
      shard: default_shard(),
      current_vpks: Vec::new(),
      disabled_vpks: Vec::new(),
      original_vpk_names: Vec::new(),
    }
  }
}

impl ProfileVpkManifestEntry {
  pub fn file_paths(&self, base: &ProfileBase) -> Vec<std::path::PathBuf> {
    if self.enabled {
      let enabled_dir = base.shard_dir(self.shard);
      self
        .current_vpks
        .iter()
        .map(|vpk| enabled_dir.join(vpk))
        .collect()
    } else {
      self
        .disabled_vpks
        .iter()
        .map(|vpk| base.join(vpk))
        .collect()
    }
  }
}

impl Default for ProfileVpkManifest {
  fn default() -> Self {
    Self {
      version: CURRENT_MANIFEST_VERSION,
      mods: BTreeMap::new(),
    }
  }
}

impl ProfileVpkManifest {
  /// Validate every manifest-backed profile in a shard-root snapshot, returning
  /// the first problem of each invalid profile keyed by its folder relative to
  /// `addons` (empty for the default profile).
  ///
  /// `root` contains sibling `addons`, `addons2`, ... directories, as used by
  /// both the live `citadel` tree and addons backups.
  pub fn profile_failures(root: &Path) -> Result<BTreeMap<String, Error>, Error> {
    let addons_root = root.join("addons");
    if !addons_root.is_dir() {
      return Err(Error::ModInvalid(format!(
        "Snapshot has no addons directory at {}",
        addons_root.display()
      )));
    }

    let mut failures = BTreeMap::new();
    let mut pending = vec![addons_root.clone()];
    while let Some(profile_path) = pending.pop() {
      for entry in fs::read_dir(&profile_path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
          pending.push(entry.path());
        }
      }

      if !profile_path.join(MANIFEST_FILENAME).is_file() {
        continue;
      }
      let profile_folder = profile_path
        .strip_prefix(&addons_root)
        .unwrap_or(&profile_path)
        .to_string_lossy()
        .replace('\\', "/");
      if let Err(error) = Self::validate_profile(root, &profile_path, &profile_folder) {
        failures.insert(profile_folder, error);
      }
    }

    Ok(failures)
  }

  #[cfg(test)]
  pub fn validate_tree(root: &Path) -> Result<(), Error> {
    match Self::profile_failures(root)?.into_values().next() {
      Some(error) => Err(error),
      None => Ok(()),
    }
  }

  fn validate_profile(root: &Path, profile_path: &Path, profile_folder: &str) -> Result<(), Error> {
    let base = ProfileBase::from_snapshot(profile_path)?;
    let manifest = Self::load(profile_path)?;
    let mut claimed = BTreeMap::<String, String>::new();
    for (mod_id, entry) in &manifest.mods {
      let missing: Vec<String> = entry
        .file_paths(&base)
        .into_iter()
        .filter(|path| !path.is_file())
        .map(|path| path.display().to_string())
        .collect();
      if !missing.is_empty() {
        return Err(Error::ModInvalid(format!(
          "Manifest entry {mod_id} references missing VPKs: {}",
          missing.join(", ")
        )));
      }
      for path in entry.file_paths(&base) {
        let key = path.display().to_string();
        if let Some(owner) = claimed.insert(key, mod_id.clone()) {
          return Err(Error::DuplicateVpkOwnership {
            mod_ids: vec![owner, mod_id.clone()],
            path: path
              .strip_prefix(root)
              .unwrap_or(&path)
              .to_string_lossy()
              .replace('\\', "/"),
            profile_folder: profile_folder.to_string(),
          });
        }
      }
    }
    Ok(())
  }

  pub fn shard_of(&self, mod_id: &str) -> ShardIndex {
    self
      .mods
      .get(mod_id)
      .map(|entry| entry.shard)
      .unwrap_or(ShardIndex::FIRST)
  }

  /// Read a profile's manifest without touching anything on disk.
  ///
  /// A crashed [`Self::save`] can leave the manifest only in the temp file; that
  /// case is resolved in memory here so reads see the committed data, and the
  /// files are reconciled by [`Self::open_for_write`].
  pub fn load(addons_path: &Path) -> Result<Self, Error> {
    let manifest_path = addons_path.join(MANIFEST_FILENAME);
    let temp_path = addons_path.join(MANIFEST_TEMP_FILENAME);

    let source_path = if manifest_path.exists() {
      Some(manifest_path.clone())
    } else if temp_path.exists() {
      Some(temp_path)
    } else {
      None
    };

    let manifest: Self = match source_path {
      Some(path) => {
        let manifest_json = fs::read_to_string(&path)?;
        serde_json::from_str(&manifest_json).map_err(|e| {
          Error::InvalidInput(format!(
            "Failed to parse VPK manifest at {}: {e}",
            path.display()
          ))
        })?
      }
      None => Self::default(),
    };

    manifest.validate_version(addons_path)
  }

  pub(crate) fn validate_version(mut self, addons_path: &Path) -> Result<Self, Error> {
    if self.version > CURRENT_MANIFEST_VERSION {
      return Err(Error::InvalidInput(format!(
        "VPK manifest at {} uses unsupported version {}",
        addons_path.join(MANIFEST_FILENAME).display(),
        self.version
      )));
    }

    if self.version < CURRENT_MANIFEST_VERSION {
      self.version = CURRENT_MANIFEST_VERSION;
    }

    Ok(self)
  }

  /// Read a profile's manifest and repair whatever a crashed operation left
  /// behind: a half-committed manifest temp file and any staged VPKs still
  /// parked in a staging directory.
  ///
  /// Only mutation entry points may call this. Reads must use [`Self::load`],
  /// which never writes -- notably [`Self::profile_failures`], which walks backup
  /// snapshots that must come out byte-identical.
  pub fn open_for_write(addons_path: &Path) -> Result<ManifestWriteGuard, Error> {
    let profile_lock = lock_profile(addons_path)?;
    let manifest = Self::open_for_write_under_lock(addons_path)?;
    Ok(ManifestWriteGuard {
      manifest,
      _profile_lock: profile_lock,
    })
  }

  /// Same repair-and-read cycle for a caller that already holds the profile
  /// lock, e.g. one that keeps several saves inside a single critical section.
  ///
  /// Taking [`lock_profile`] twice on one thread is a lost update by
  /// construction, so it is rejected rather than nested; this is the escape
  /// hatch for the legitimate case.
  pub fn open_for_write_under_lock(addons_path: &Path) -> Result<ProfileVpkManifest, Error> {
    crate::mod_manager::file_permissions::prepare_profile_for_write(addons_path)?;
    Self::reconcile_manifest_temp_file(addons_path)?;
    let manifest = Self::load(addons_path)?;
    Self::recover_pending_staging(addons_path, &manifest)?;
    Ok(manifest)
  }

  /// Repair a profile in place without keeping the manifest around, for entry
  /// points that mutate the profile without reading it.
  pub fn recover_profile(addons_path: &Path) -> Result<(), Error> {
    Self::open_for_write(addons_path).map(|_| ())
  }

  fn reconcile_manifest_temp_file(addons_path: &Path) -> Result<(), Error> {
    let manifest_path = addons_path.join(MANIFEST_FILENAME);

    // A commit is a fsync'ed write followed by a rename, so a temp file is a
    // complete manifest. If the main file is gone, the newest temp file is the
    // newest state we have and is promoted rather than deleted.
    Self::recover_newest_save_temp(addons_path, &manifest_path);
    Self::remove_stale_manifest_temps(addons_path);

    let temp_path = addons_path.join(MANIFEST_TEMP_FILENAME);
    if !temp_path.exists() {
      return Ok(());
    }

    if manifest_path.exists() {
      fs::remove_file(&temp_path).map_err(|e| {
        Error::InvalidInput(format!(
          "Failed to remove stale VPK manifest temp file at {}: {e}",
          temp_path.display()
        ))
      })?;
    } else {
      fs::rename(&temp_path, &manifest_path).map_err(|e| {
        Error::InvalidInput(format!(
          "Failed to recover VPK manifest temp file from {} to {}: {e}",
          temp_path.display(),
          manifest_path.display()
        ))
      })?;
    }

    Ok(())
  }

  fn recover_pending_staging(addons_path: &Path, manifest: &Self) -> Result<(), Error> {
    let base = ProfileBase::from_snapshot(addons_path)?;
    let clear_staging = addons_path.join(CLEAR_STAGING_DIR);
    if clear_staging.is_dir() {
      let mode = if manifest.mods.is_empty() {
        RecoveryMode::Discard
      } else {
        RecoveryMode::Strict
      };
      recover_staging_directory(&base, &clear_staging, mode, manifest)?;
    }

    let reorder_staging = addons_path.join(REORDER_STAGING_DIR);
    if reorder_staging.is_dir() {
      recover_staging_directory(
        &base,
        &reorder_staging,
        RecoveryMode::AvoidEnabledCollisions,
        manifest,
      )?;
    }

    if !addons_path.is_dir() {
      return Ok(());
    }
    for entry in fs::read_dir(addons_path)? {
      let entry = entry?;
      let path = entry.path();
      let entry_name = entry.file_name();
      let Some(mod_id) = entry_name
        .to_str()
        .and_then(|name| name.strip_prefix(UPDATE_STAGING_PREFIX))
        .filter(|_| path.is_dir())
      else {
        continue;
      };
      let mode = if manifest.mods.contains_key(mod_id) {
        RecoveryMode::Strict
      } else {
        RecoveryMode::Discard
      };
      recover_staging_directory(&base, &path, mode, manifest)?;
    }
    legacy_reorder::recover(&base)
  }

  pub fn save(&self, addons_path: &Path) -> Result<(), Error> {
    fs::create_dir_all(addons_path)?;

    let manifest_path = addons_path.join(MANIFEST_FILENAME);
    // One temp file per save, never a shared `.dmm.json.tmp`: two saves on the
    // same profile must not overwrite each other's file, and a concurrent
    // `open_for_write` must not delete a commit that is still in flight.
    let temp_path = next_manifest_temp_path(addons_path);
    let manifest_json = serde_json::to_string_pretty(self).map_err(|e| {
      Error::InvalidInput(format!(
        "Failed to serialize VPK manifest at {}: {e}",
        manifest_path.display()
      ))
    })?;

    if let Err(error) = Self::commit_manifest(&manifest_json, &temp_path, &manifest_path) {
      // The temp file holds the only complete copy when the rename failed, so
      // it is kept whenever the manifest itself did not make it to disk: the
      // next `open_for_write` promotes it back.
      if manifest_path.exists() {
        let _ = fs::remove_file(&temp_path);
      }
      return Err(error);
    }

    Ok(())
  }

  fn commit_manifest(
    manifest_json: &str,
    temp_path: &Path,
    manifest_path: &Path,
  ) -> Result<(), Error> {
    crate::mod_manager::file_permissions::ensure_file_writable(manifest_path)?;
    crate::mod_manager::file_permissions::ensure_file_writable(temp_path)?;
    // Flush before the rename: NTFS journals the rename and file size, so a
    // power loss after an unsynced write leaves a manifest of NUL bytes.
    let mut temp_file = fs::File::create(temp_path)?;
    temp_file.write_all(manifest_json.as_bytes())?;
    temp_file.sync_all()?;
    drop(temp_file);
    if let Err(err) = fs::rename(temp_path, manifest_path) {
      // Rust maps `fs::rename` to `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` on
      // Windows, so an existing target does not normally surface at all. Keep a
      // fallback anyway, but never delete the manifest and hope: the old one is
      // parked next to the new and restored if the swap fails, so no sequence of
      // failures can lose both copies.
      if err.kind() != ErrorKind::AlreadyExists {
        return Err(err.into());
      }
      Self::swap_manifest_over_existing(temp_path, manifest_path)?;
    }

    Ok(())
  }

  fn swap_manifest_over_existing(temp_path: &Path, manifest_path: &Path) -> Result<(), Error> {
    let backup_path = manifest_path.with_file_name(format!(
      "{MANIFEST_TEMP_PREFIX}replaced-{}",
      SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0)
    ));
    fs::rename(manifest_path, &backup_path)?;

    match fs::rename(temp_path, manifest_path) {
      Ok(()) => {
        if let Err(error) = fs::remove_file(&backup_path) {
          log::warn!(
            "Replaced VPK manifest but failed to remove its backup at {}: {error}",
            backup_path.display()
          );
        }
        Ok(())
      }
      Err(swap_error) => {
        if let Err(restore_error) = fs::rename(&backup_path, manifest_path) {
          log::error!(
            "Failed to restore VPK manifest from {} after a failed swap ({swap_error}): {restore_error}",
            backup_path.display()
          );
        }
        Err(swap_error.into())
      }
    }
  }

  /// Promote the newest per-save temp file when the manifest is missing.
  ///
  /// Covers both a save that crashed between fsync and rename and a failed
  /// [`Self::swap_manifest_over_existing`]: the temp file is a complete
  /// manifest, so leaving it behind as garbage would throw away the newest
  /// state the profile has.
  fn recover_newest_save_temp(addons_path: &Path, manifest_path: &Path) {
    if manifest_path.exists() {
      return;
    }

    let entries = match fs::read_dir(addons_path) {
      Ok(entries) => entries,
      Err(_) => return,
    };

    let mut newest: Option<(SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
      let file_name = entry.file_name();
      let Some(name) = file_name.to_str() else {
        continue;
      };
      // The legacy name is handled separately below, and a non-zero length is
      // required because a temp file is only complete after `sync_all`.
      if name == MANIFEST_TEMP_FILENAME || temp_path_pid(name).is_none() {
        continue;
      }
      let Ok(metadata) = entry.metadata() else {
        continue;
      };
      if metadata.len() == 0 {
        continue;
      }
      let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
      if newest.as_ref().is_none_or(|(seen, _)| modified > *seen) {
        newest = Some((modified, entry.path()));
      }
    }

    if let Some((_, path)) = newest {
      if let Err(error) = fs::rename(&path, manifest_path) {
        log::warn!(
          "Failed to recover VPK manifest from {} to {}: {error}",
          path.display(),
          manifest_path.display()
        );
      } else {
        log::warn!(
          "Recovered VPK manifest at {} from an uncommitted temp file",
          manifest_path.display()
        );
      }
    }
  }

  /// Drop manifest temp files left behind by a crashed save.
  ///
  /// The profile write lock only serializes writers in this process, so a temp
  /// file's name -- the pid of its writer -- and age decide whether it can go: a
  /// live peer may be committing right now and must not lose its rename.
  fn remove_stale_manifest_temps(addons_path: &Path) {
    let entries = match fs::read_dir(addons_path) {
      Ok(entries) => entries,
      Err(_) => return,
    };

    let own_pid = std::process::id();
    for entry in entries.flatten() {
      let file_name = entry.file_name();
      let Some(name) = file_name.to_str() else {
        continue;
      };

      // The legacy name is reconciled by the caller: its content may be the
      // only copy of the manifest, so it is never deleted here.
      if name == MANIFEST_TEMP_FILENAME {
        continue;
      }
      let pid = temp_path_pid(name);
      if pid.is_none() {
        continue;
      }

      let Ok(metadata) = entry.metadata() else {
        continue;
      };
      let path = entry.path();
      let age = metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .unwrap_or_default();
      let writer_is_alive = pid.is_some_and(process_is_alive);
      if !should_remove_temp_file(pid, own_pid, age, writer_is_alive) {
        continue;
      }

      if let Err(error) = fs::remove_file(&path) {
        log::debug!(
          "Failed to remove stale VPK manifest temp file {}: {error}",
          path.display()
        );
      }
    }
  }

  pub fn mark_enabled(
    &mut self,
    mod_id: &str,
    current_vpks: Vec<String>,
    original_vpk_names: Vec<String>,
    order: Option<u32>,
    shard: ShardIndex,
  ) {
    let entry = self.mods.entry(mod_id.to_string()).or_default();
    entry.enabled = true;
    entry.shard = shard;
    entry.current_vpks = current_vpks;
    entry.disabled_vpks.clear();
    if !original_vpk_names.is_empty() {
      entry.original_vpk_names = original_vpk_names;
    }
    if order.is_some() {
      entry.order = order;
    }
  }

  pub fn mark_disabled(
    &mut self,
    mod_id: &str,
    disabled_vpks: Vec<String>,
    original_vpk_names: Vec<String>,
  ) {
    let entry = self.mods.entry(mod_id.to_string()).or_default();
    entry.enabled = false;
    entry.shard = default_shard();
    entry.current_vpks.clear();
    entry.disabled_vpks = disabled_vpks;
    if !original_vpk_names.is_empty() {
      entry.original_vpk_names = original_vpk_names;
    }
  }

  /// Remove aliases of files deleted by a mutation, including old duplicate claims.
  pub fn forget_files(&mut self, base: &ProfileBase, removed: &[std::path::PathBuf]) {
    for entry in self.mods.values_mut() {
      let paths = entry.file_paths(base);
      if !paths.iter().any(|path| removed.contains(path)) {
        continue;
      }
      let names = if entry.enabled {
        &mut entry.current_vpks
      } else {
        &mut entry.disabled_vpks
      };
      let originals = std::mem::take(&mut entry.original_vpk_names);
      let aligned = originals.len() == names.len();
      let mut index = 0;
      names.retain(|_| {
        let keep = !removed.contains(&paths[index]);
        if keep && aligned {
          entry.original_vpk_names.push(originals[index].clone());
        }
        index += 1;
        keep
      });
      if entry.enabled && names.is_empty() {
        entry.enabled = false;
      }
    }
  }

  pub fn remove_mod(&mut self, mod_id: &str) {
    self.mods.remove(mod_id);
  }

  pub fn migrate_mod_identity(&mut self, from: &str, to: &str) -> Result<(), Error> {
    if from == to {
      return Ok(());
    }

    if self.mods.contains_key(from) && self.mods.contains_key(to) {
      if self.mods.get(from) != self.mods.get(to) {
        return Err(Error::ModInvalid(format!(
          "VPK manifest contains conflicting entries for {from} and {to}"
        )));
      }
      self.mods.remove(from);
    } else if let Some(entry) = self.mods.remove(from) {
      self.mods.insert(to.to_string(), entry);
    }

    if let Some(entry) = self.mods.get_mut(to) {
      for file_name in &mut entry.disabled_vpks {
        if let Some(suffix) = file_name.strip_prefix(&format!("{from}_")) {
          *file_name = format!("{to}_{suffix}");
        }
      }
    }
    self.version = CURRENT_MANIFEST_VERSION;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn addons_base(temp: &tempfile::TempDir) -> std::path::PathBuf {
    let base = temp.path().join("citadel").join("addons");
    fs::create_dir_all(&base).unwrap();
    base
  }

  /// Byte-for-byte what the last pre-sharding release wrote: version 1, and no
  /// `shard` key anywhere. Every existing install has a file shaped like this,
  /// so upgrading it is the one path that cannot be allowed to break.
  const V1_MANIFEST: &str = r#"{
  "version": 1,
  "mods": {
    "123456": {
      "enabled": true,
      "order": 0,
      "currentVpks": [
        "pak01_dir.vpk",
        "pak02_dir.vpk"
      ],
      "disabledVpks": [],
      "originalVpkNames": [
        "cool_mod.vpk",
        "cool_mod_2.vpk"
      ]
    },
    "local-abc": {
      "enabled": false,
      "order": 3,
      "currentVpks": [],
      "disabledVpks": [
        "local-abc_skin.vpk"
      ],
      "originalVpkNames": [
        "skin.vpk"
      ]
    }
  }
}"#;

  #[test]
  fn upgrades_a_v1_manifest_from_disk_to_the_current_version() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    fs::write(base.join(MANIFEST_FILENAME), V1_MANIFEST).unwrap();

    let loaded = ProfileVpkManifest::load(&base).unwrap();

    assert_eq!(loaded.version, CURRENT_MANIFEST_VERSION);
    assert_eq!(loaded.mods.len(), 2);

    // Pre-sharding installs kept every enabled VPK in the profile base.
    let enabled = loaded.mods.get("123456").unwrap();
    assert_eq!(enabled.shard, ShardIndex::FIRST);
    assert!(enabled.enabled);
    assert_eq!(enabled.order, Some(0));
    assert_eq!(
      enabled.current_vpks,
      vec!["pak01_dir.vpk".to_string(), "pak02_dir.vpk".to_string()]
    );
    assert_eq!(
      enabled.original_vpk_names,
      vec!["cool_mod.vpk".to_string(), "cool_mod_2.vpk".to_string()]
    );

    let disabled = loaded.mods.get("local-abc").unwrap();
    assert_eq!(disabled.shard, ShardIndex::FIRST);
    assert!(!disabled.enabled);
    assert_eq!(
      disabled.disabled_vpks,
      vec!["local-abc_skin.vpk".to_string()]
    );

    // A v1 entry resolves against the base directory, exactly where its files are.
    let profile = ProfileBase::new(&base).unwrap();
    assert_eq!(
      enabled.file_paths(&profile),
      vec![base.join("pak01_dir.vpk"), base.join("pak02_dir.vpk")]
    );
  }

  /// The upgrade is only in memory until something saves; the next write must
  /// persist it so the file stops being re-upgraded on every load.
  #[test]
  fn saving_an_upgraded_v1_manifest_writes_the_current_version() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    fs::write(base.join(MANIFEST_FILENAME), V1_MANIFEST).unwrap();

    let loaded = ProfileVpkManifest::load(&base).unwrap();
    loaded.save(&base).unwrap();

    let written = fs::read_to_string(base.join(MANIFEST_FILENAME)).unwrap();
    assert!(written.contains(&format!("\"version\": {CURRENT_MANIFEST_VERSION}")));
    assert!(written.contains("\"shard\": 1"));
    assert_eq!(ProfileVpkManifest::load(&base).unwrap(), loaded);
  }

  #[test]
  fn v3_manifest_migrates_sound_keys_and_disabled_file_names_idempotently() {
    let mut manifest = ProfileVpkManifest {
      version: 2,
      mods: BTreeMap::from([(
        "42".to_string(),
        ProfileVpkManifestEntry {
          disabled_vpks: vec!["42_voice.vpk".to_string()],
          ..ProfileVpkManifestEntry::default()
        },
      )]),
    };

    manifest.migrate_mod_identity("42", "snd-42").unwrap();
    manifest.migrate_mod_identity("42", "snd-42").unwrap();

    assert_eq!(manifest.version, CURRENT_MANIFEST_VERSION);
    assert!(!manifest.mods.contains_key("42"));
    assert_eq!(
      manifest.mods["snd-42"].disabled_vpks,
      vec!["snd-42_voice.vpk"]
    );
  }

  #[test]
  fn identity_migration_rejects_conflicting_manifest_entries() {
    let mut manifest = ProfileVpkManifest {
      version: 2,
      mods: BTreeMap::from([
        ("42".to_string(), ProfileVpkManifestEntry::default()),
        (
          "snd-42".to_string(),
          ProfileVpkManifestEntry {
            enabled: true,
            ..ProfileVpkManifestEntry::default()
          },
        ),
      ]),
    };

    assert!(manifest.migrate_mod_identity("42", "snd-42").is_err());
  }

  /// A manifest written by a newer build must not be silently mis-parsed into
  /// a shard layout this build does not understand.
  #[test]
  fn load_rejects_a_manifest_from_a_future_version() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    fs::write(
      base.join(MANIFEST_FILENAME),
      format!(
        r#"{{"version": {}, "mods": {{}}}}"#,
        CURRENT_MANIFEST_VERSION + 1
      ),
    )
    .unwrap();

    let error = ProfileVpkManifest::load(&base).unwrap_err();

    assert!(error.to_string().contains("unsupported version"));
  }

  /// Shard indexes outside `1..=MAX_SHARDS` cannot address a real directory, so
  /// a corrupted entry has to fail loudly rather than resolve to shard 1.
  #[test]
  fn load_rejects_an_out_of_range_shard() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    fs::write(
      base.join(MANIFEST_FILENAME),
      r#"{"version": 2, "mods": {"1": {"enabled": true, "shard": 0}}}"#,
    )
    .unwrap();

    assert!(ProfileVpkManifest::load(&base).is_err());
  }

  #[test]
  fn persists_profile_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );

    manifest.save(&base).unwrap();
    let loaded = ProfileVpkManifest::load(&base).unwrap();

    assert_eq!(loaded, manifest);
  }

  #[test]
  fn open_for_write_recovers_temp_manifest_when_main_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );
    let temp_path = base.join(format!("{MANIFEST_FILENAME}.tmp"));
    fs::write(&temp_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    // A read sees the committed data without moving the file...
    assert_eq!(ProfileVpkManifest::load(&base).unwrap(), manifest);
    assert!(temp_path.exists());

    // ...and the write path is what reconciles it.
    let loaded = ProfileVpkManifest::open_for_write(&base).unwrap();

    assert_eq!(loaded, manifest);
    assert!(base.join(MANIFEST_FILENAME).exists());
    assert!(!temp_path.exists());
  }

  #[test]
  fn open_for_write_removes_stale_temp_manifest_when_main_exists() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );
    manifest.save(&base).unwrap();
    let temp_path = base.join(format!("{MANIFEST_FILENAME}.tmp"));
    fs::write(&temp_path, "{}").unwrap();

    let loaded = ProfileVpkManifest::open_for_write(&base).unwrap();

    assert_eq!(loaded, manifest);
    assert!(!temp_path.exists());
  }

  #[test]
  fn open_for_write_recovers_interrupted_reorder_staging() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let staging = base.join(REORDER_STAGING_DIR);
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("s2__pak01_dir.vpk.pending"), b"vpk").unwrap();

    ProfileVpkManifest::open_for_write(&base).unwrap();

    assert_eq!(
      fs::read(temp.path().join("citadel/addons2/pak01_dir.vpk")).unwrap(),
      b"vpk"
    );
    assert!(!staging.exists());
  }

  /// Reads must never repair. Recovery relocates files across shards, so a read
  /// that triggered it would mutate whatever tree it was pointed at.
  #[test]
  fn load_leaves_interrupted_staging_alone() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let staging = base.join(REORDER_STAGING_DIR);
    fs::create_dir_all(&staging).unwrap();
    let staged = staging.join("s2__pak01_dir.vpk.pending");
    fs::write(&staged, b"vpk").unwrap();

    ProfileVpkManifest::load(&base).unwrap();

    assert_eq!(fs::read(&staged).unwrap(), b"vpk");
    assert!(!temp.path().join("citadel/addons2/pak01_dir.vpk").exists());
  }

  /// Backups are validated by walking the snapshot. A snapshot taken mid-operation
  /// contains staging directories by construction, and validating it must not
  /// relocate or delete their contents.
  #[test]
  fn validate_tree_does_not_mutate_the_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let snapshot = temp.path().join("snapshot");
    let profile = snapshot.join("addons/profile_x");
    fs::create_dir_all(&profile).unwrap();

    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );
    manifest.save(&profile).unwrap();
    fs::write(profile.join("pak01_dir.vpk"), b"vpk").unwrap();

    let staging = profile.join(REORDER_STAGING_DIR);
    fs::create_dir_all(&staging).unwrap();
    let staged = staging.join("s2__pak02_dir.vpk.pending");
    fs::write(&staged, b"staged vpk").unwrap();

    let orphan_staging = profile.join(format!("{UPDATE_STAGING_PREFIX}999"));
    fs::create_dir_all(&orphan_staging).unwrap();
    let orphan_staged = orphan_staging.join("s1__pak03_dir.vpk.pending");
    fs::write(&orphan_staged, b"orphan vpk").unwrap();

    ProfileVpkManifest::validate_tree(&snapshot).unwrap();

    assert_eq!(fs::read(&staged).unwrap(), b"staged vpk");
    assert_eq!(fs::read(&orphan_staged).unwrap(), b"orphan vpk");
    assert!(
      !snapshot.join("addons2/profile_x/pak02_dir.vpk").exists(),
      "validation relocated a staged VPK inside the backup"
    );
  }

  #[test]
  fn validate_tree_reports_duplicate_owners_with_snapshot_relative_paths() {
    for folder in ["", "profile_other"] {
      let temp = tempfile::tempdir().unwrap();
      let snapshot = temp.path().join("snapshot");
      let profile = snapshot.join("addons").join(folder);
      let shard = snapshot.join("addons2").join(folder);
      fs::create_dir_all(&profile).unwrap();
      fs::create_dir_all(&shard).unwrap();
      fs::write(shard.join("pak19_dir.vpk"), b"vpk").unwrap();
      let mut manifest = ProfileVpkManifest::default();
      for id in ["636266", "658671"] {
        manifest.mark_enabled(
          id,
          vec!["pak19_dir.vpk".into()],
          vec![],
          Some(0),
          ShardIndex::new(2).unwrap(),
        );
      }
      manifest.save(&profile).unwrap();
      let before = fs::read(profile.join(MANIFEST_FILENAME)).unwrap();
      let error = ProfileVpkManifest::validate_tree(&snapshot).unwrap_err();
      let json = serde_json::to_value(error).unwrap();
      assert_eq!(json["kind"], "duplicateVpkOwnership");
      assert_eq!(
        json["conflictingMods"],
        serde_json::json!(["636266", "658671"])
      );
      assert_eq!(json["profileFolder"], folder);
      assert_eq!(
        json["path"],
        if folder.is_empty() {
          "addons2/pak19_dir.vpk".to_string()
        } else {
          format!("addons2/{folder}/pak19_dir.vpk")
        }
      );
      assert_eq!(fs::read(profile.join(MANIFEST_FILENAME)).unwrap(), before);
      assert_eq!(fs::read(shard.join("pak19_dir.vpk")).unwrap(), b"vpk");
    }
  }

  #[test]
  fn validate_tree_checks_manifest_files_in_their_recorded_shards() {
    let temp = tempfile::tempdir().unwrap();
    let snapshot = temp.path().join("snapshot");
    let profile = snapshot.join("addons/profile_x");
    let shard_two = snapshot.join("addons2/profile_x");
    fs::create_dir_all(&profile).unwrap();

    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::new(2).unwrap(),
    );
    manifest.save(&profile).unwrap();

    let error = ProfileVpkManifest::validate_tree(&snapshot).unwrap_err();
    assert!(error.to_string().contains("references missing VPKs"));

    fs::create_dir_all(&shard_two).unwrap();
    fs::write(shard_two.join("pak01_dir.vpk"), b"vpk").unwrap();
    ProfileVpkManifest::validate_tree(&snapshot).unwrap();
  }

  #[test]
  fn deleting_shared_file_preserves_other_shards_and_remaining_name_mappings() {
    let temp = tempfile::tempdir().unwrap();
    let base_path = addons_base(&temp);
    let base = ProfileBase::from_snapshot(&base_path).unwrap();
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "alias",
      vec!["pak01_dir.vpk".into(), "pak02_dir.vpk".into()],
      vec!["first.vpk".into(), "second.vpk".into()],
      None,
      ShardIndex::FIRST,
    );
    manifest.mark_enabled(
      "other-shard",
      vec!["pak01_dir.vpk".into()],
      vec!["unrelated.vpk".into()],
      None,
      ShardIndex::new(2).unwrap(),
    );
    manifest.forget_files(&base, &[base.join("pak01_dir.vpk")]);
    assert_eq!(manifest.mods["alias"].current_vpks, ["pak02_dir.vpk"]);
    assert_eq!(manifest.mods["alias"].original_vpk_names, ["second.vpk"]);
    assert!(manifest.mods["alias"].enabled);
    assert_eq!(manifest.mods["other-shard"].current_vpks, ["pak01_dir.vpk"]);
  }

  #[test]
  fn mark_enabled_preserves_original_names_when_empty() {
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );

    manifest.mark_enabled(
      "123",
      vec!["pak02_dir.vpk".to_string()],
      Vec::new(),
      Some(1),
      ShardIndex::new(2).unwrap(),
    );

    let entry = manifest.mods.get("123").unwrap();
    assert_eq!(entry.original_vpk_names, vec!["cool_mod.vpk".to_string()]);
    assert_eq!(entry.current_vpks, vec!["pak02_dir.vpk".to_string()]);
    assert_eq!(entry.order, Some(1));
    assert_eq!(entry.shard, ShardIndex::new(2).unwrap());
  }
  #[test]
  fn compatibility_defaults_off_and_survives_mod_enable_disable_and_reorder() {
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "mod",
      vec!["pak01_dir.vpk".into()],
      vec![],
      Some(1),
      ShardIndex::FIRST,
    );
    assert!(!manifest.mods["mod"].compatibility_enabled);
    manifest.mods.get_mut("mod").unwrap().compatibility_enabled = true;
    manifest
      .mods
      .get_mut("mod")
      .unwrap()
      .compatibility_opted_out = true;
    manifest.mark_disabled("mod", vec!["mod_source.vpk".into()], vec![]);
    manifest.mark_enabled(
      "mod",
      vec!["pak02_dir.vpk".into()],
      vec![],
      Some(2),
      ShardIndex::FIRST,
    );
    assert!(manifest.mods["mod"].compatibility_enabled);
    let encoded = serde_json::to_vec(&manifest).unwrap();
    let restored: ProfileVpkManifest = serde_json::from_slice(&encoded).unwrap();
    assert!(restored.mods["mod"].compatibility_enabled);
    assert!(restored.mods["mod"].compatibility_opted_out);
    let old: ProfileVpkManifestEntry = serde_json::from_str(r#"{"enabled":true}"#).unwrap();
    assert!(!old.compatibility_enabled);
    assert!(!old.compatibility_opted_out);
  }

  /// Saves must not share one temp name: a second writer truncating the first
  /// writer's file is what turned a reinstall into "No such file or directory".
  #[test]
  fn every_save_uses_its_own_temp_file_name() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );

    manifest.save(&base).unwrap();

    let leftovers: Vec<String> = fs::read_dir(&base)
      .unwrap()
      .flatten()
      .map(|entry| entry.file_name().to_string_lossy().to_string())
      .filter(|name| crate::mod_manager::shard::is_manifest_temp_name(name))
      .collect();
    assert!(
      leftovers.is_empty(),
      "a committed save must not leave temp files: {leftovers:?}"
    );
    assert_eq!(ProfileVpkManifest::load(&base).unwrap(), manifest);

    let first = next_manifest_temp_path(&base);
    let second = next_manifest_temp_path(&base);
    assert_ne!(first, second);
    assert!(crate::mod_manager::shard::is_internal_artifact(
      first.file_name().unwrap().to_str().unwrap()
    ));
  }

  /// A crashed save leaves a uniquely named temp file; the write path owns the
  /// profile at that point, so it can and should clear it.
  #[test]
  fn open_for_write_removes_stale_unique_temp_files() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );
    manifest.save(&base).unwrap();
    // Named with this process's pid: the profile lock is held, so a temp file
    // of ours cannot be a commit in flight.
    let stale = base.join(format!(".dmm.json.{}-1-1.tmp", std::process::id()));
    fs::write(&stale, "{}").unwrap();

    let loaded = ProfileVpkManifest::open_for_write(&base).unwrap();

    assert_eq!(loaded, manifest);
    assert!(
      !stale.exists(),
      "open_for_write must drop temp files no writer owns"
    );
  }

  /// Another process may be committing right now: its temp file must survive,
  /// because deleting it turns its rename into `ENOENT` -- the reported bug.
  #[test]
  fn temp_file_of_a_live_peer_is_left_alone() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    ProfileVpkManifest::default().save(&base).unwrap();
    let peer_pid = std::process::id() + 1;
    let peer_temp = base.join(format!(".dmm.json.{peer_pid}-1-1.tmp"));
    fs::write(&peer_temp, "{}").unwrap();

    let keep = should_remove_temp_file(Some(peer_pid), std::process::id(), Duration::ZERO, true);
    let reclaim = should_remove_temp_file(
      Some(peer_pid),
      std::process::id(),
      MANIFEST_TEMP_TTL + Duration::from_secs(1),
      true,
    );
    assert!(!keep, "a live writer's temp file must not be deleted");
    assert!(
      reclaim,
      "an abandoned temp file must eventually be reclaimed"
    );

    // Our own leftovers, and those of a process that is gone, are reclaimable
    // immediately.
    assert!(should_remove_temp_file(
      Some(std::process::id()),
      std::process::id(),
      Duration::ZERO,
      true
    ));
    assert!(should_remove_temp_file(
      Some(peer_pid),
      std::process::id(),
      Duration::ZERO,
      false
    ));
    assert_eq!(temp_path_pid(".dmm.json.4242-7-3.tmp"), Some(4242));
    assert_eq!(temp_path_pid(".dmm.json.tmp"), None);
    assert_eq!(temp_path_pid(".dmm.json"), None);
  }

  /// A save that crashed between `sync_all` and the rename, or a failed swap,
  /// leaves the newest complete manifest in a temp file. Deleting it would
  /// throw away the newest state the profile has.
  #[test]
  fn open_for_write_recovers_newest_save_temp_when_manifest_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);
    let mut manifest = ProfileVpkManifest::default();
    manifest.mark_enabled(
      "123",
      vec!["pak01_dir.vpk".to_string()],
      vec!["cool_mod.vpk".to_string()],
      Some(0),
      ShardIndex::FIRST,
    );
    manifest.save(&base).unwrap();
    let manifest_path = base.join(MANIFEST_FILENAME);
    let crashed = base.join(format!(".dmm.json.{}-1-1.tmp", std::process::id()));
    fs::rename(&manifest_path, &crashed).unwrap();
    // An older, empty remnant must not win over the complete copy.
    fs::write(
      base.join(format!(".dmm.json.{}-0-0.tmp", std::process::id())),
      "",
    )
    .unwrap();

    let recovered = ProfileVpkManifest::open_for_write(&base).unwrap();

    assert_eq!(recovered, manifest);
    assert!(manifest_path.exists());
    assert_eq!(ProfileVpkManifest::load(&base).unwrap(), manifest);
  }

  /// The regression this fix exists for: a reinstall purges the old mod from the
  /// manifest while the download worker persists the new one. Both write, so
  /// with a shared temp name the second `open_for_write` deleted the first one's
  /// commit and the rename failed with ENOENT.
  #[test]
  fn overlapping_profile_writes_all_commit() {
    let temp = tempfile::tempdir().unwrap();
    let base = addons_base(&temp);

    let handles: Vec<_> = (0..8)
      .map(|thread| {
        let base = base.clone();
        std::thread::spawn(move || {
          for round in 0..20 {
            let mod_id = format!("mod-{thread}-{round}");
            let mut manifest = ProfileVpkManifest::open_for_write(&base).unwrap();
            manifest.mark_enabled(
              &mod_id,
              vec![format!("{mod_id}_pak01_dir.vpk")],
              Vec::new(),
              Some(round),
              ShardIndex::FIRST,
            );
            manifest.save(&base).unwrap();
          }
        })
      })
      .collect();

    for handle in handles {
      handle.join().unwrap();
    }

    // Serialized read-modify-write means no update is lost, not merely that no
    // call fails.
    let committed = ProfileVpkManifest::load(&base).unwrap();
    for thread in 0..8 {
      for round in 0..20 {
        let mod_id = format!("mod-{thread}-{round}");
        assert!(
          committed.mods.contains_key(&mod_id),
          "{mod_id} was lost by a concurrent writer"
        );
      }
    }
  }
}
