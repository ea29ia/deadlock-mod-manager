//! Serialize read-modify-write cycles on a single VPK profile.
//!
//! [`ProfileVpkManifest::save`](crate::mod_manager::vpk_manifest::ProfileVpkManifest::save)
//! commits through a temp file and a rename, and
//! [`open_for_write`](crate::mod_manager::vpk_manifest::ProfileVpkManifest::open_for_write)
//! reconciles temp files it finds. Two operations that overlap on the same
//! profile used to destroy each other's commit, and the loser's rename failed
//! with `ENOENT` -- reported to the user as the unhelpful "No such file or
//! directory (os error 2)".
//!
//! That overlap is not hypothetical: reinstalling a mod purges its old VPKs
//! (removing them from the manifest) while the download worker persists the
//! freshly placed ones, and the save interrupted in between is exactly the one
//! whose rename then failed. The window is not tiny either, because the
//! committing save flushes to disk before renaming.
//!
//! Every profile directory therefore gets a mutex that
//! [`ManifestWriteGuard`](crate::mod_manager::vpk_manifest::ManifestWriteGuard)
//! holds for the whole read-modify-write cycle, not just for the individual
//! calls.
//!
//! The lock is deliberately *not* reentrant. A nested acquire means the outer
//! caller is holding a manifest it already modified in memory while the inner
//! one reads the version on disk, so whichever saves last silently discards the
//! other's work. [`lock_profile`] therefore reports the nesting as an error
//! instead of either deadlocking on the mutex or quietly allowing that lost
//! update; callers that legitimately already hold the lock use
//! [`ProfileVpkManifest::open_for_write_under_lock`](crate::mod_manager::vpk_manifest::ProfileVpkManifest::open_for_write_under_lock)
//! to say so.

use crate::errors::Error;
use std::{
  cell::RefCell,
  collections::{HashMap, HashSet},
  fs,
  path::{Path, PathBuf},
  sync::{Mutex, MutexGuard, OnceLock, PoisonError},
};

/// Mutex guarding one profile directory.
struct ProfileLock {
  mutex: Mutex<()>,
}

impl ProfileLock {
  fn lock(&'static self) -> MutexGuard<'static, ()> {
    // A panic while a profile write was in flight must not brick the profile
    // for the rest of the process: the manifest on disk is either the old or
    // the new one, both of which the next `open_for_write` can reconcile.
    self.mutex.lock().unwrap_or_else(PoisonError::into_inner)
  }
}

fn registry() -> &'static Mutex<HashMap<PathBuf, &'static ProfileLock>> {
  static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, &'static ProfileLock>>> = OnceLock::new();
  REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

thread_local! {
  /// Profiles this thread holds, so a nested acquire is reported as the bug it
  /// is rather than deadlocking on the non-reentrant mutex.
  static HELD_BY_THREAD: RefCell<HashSet<PathBuf>> = RefCell::new(HashSet::new());
}

/// Identity of a profile directory across `.`/`..`/symlinked spellings.
fn profile_key(profile: &Path) -> PathBuf {
  if let Ok(canonical) = fs::canonicalize(profile) {
    return canonical;
  }

  // The directory may not exist yet (a first profile write creates it), so
  // canonicalize the parent and keep the final component verbatim.
  match (profile.parent(), profile.file_name()) {
    (Some(parent), Some(name)) if !parent.as_os_str().is_empty() => {
      match fs::canonicalize(parent) {
        Ok(canonical) => canonical.join(name),
        Err(_) => profile.to_path_buf(),
      }
    }
    _ => profile.to_path_buf(),
  }
}

fn lock_for(key: &Path) -> &'static ProfileLock {
  let mut registry = registry().lock().unwrap_or_else(PoisonError::into_inner);
  // Leaked on purpose: a guard owns its lock, so the mutex has to outlive every
  // possible guard. There is one entry per profile directory the process
  // touches, which is a handful at most.
  registry.entry(key.to_path_buf()).or_insert_with(|| {
    Box::leak(Box::new(ProfileLock {
      mutex: Mutex::new(()),
    }))
  })
}

/// Holds a profile's write lock until it is dropped.
///
/// The mutex is released before the thread-local bookkeeping so the next owner
/// never sees a profile that is both held and unowned.
pub struct ProfileGuard {
  key: PathBuf,
  guard: Option<MutexGuard<'static, ()>>,
}

impl std::fmt::Debug for ProfileGuard {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter
      .debug_struct("ProfileGuard")
      .field("profile", &self.key)
      .finish()
  }
}

impl Drop for ProfileGuard {
  fn drop(&mut self) {
    drop(self.guard.take());
    HELD_BY_THREAD.with(|held| {
      held.borrow_mut().remove(&self.key);
    });
  }
}

/// Acquire the write lock for a profile directory.
///
/// [`ProfileVpkManifest::open_for_write`](crate::mod_manager::vpk_manifest::ProfileVpkManifest::open_for_write)
/// takes this lock for you; call it directly only when a longer stretch has to
/// stay serialized than one manifest read-modify-write.
///
/// Fails instead of nesting: see the module docs for why a second acquire on
/// the same thread cannot be made safe.
pub fn lock_profile(profile: &Path) -> Result<ProfileGuard, Error> {
  let key = profile_key(profile);
  let nested = HELD_BY_THREAD.with(|held| {
    let mut held = held.borrow_mut();
    if held.contains(&key) {
      true
    } else {
      held.insert(key.clone());
      false
    }
  });

  if nested {
    return Err(Error::InvalidInput(format!(
      "VPK profile {} is already write-locked by this thread; \
       nested profile writes would discard one of the two manifest changes",
      key.display()
    )));
  }

  let lock = lock_for(&key);
  Ok(ProfileGuard {
    key,
    guard: Some(lock.lock()),
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };

  #[test]
  fn same_directory_spelled_differently_shares_one_lock() {
    let temp = tempfile::tempdir().unwrap();
    let profile = temp.path().join("addons");
    fs::create_dir_all(&profile).unwrap();

    let direct = profile_key(&profile);
    let indirect = profile_key(&profile.join(".").join(""));
    let with_dotdot = profile_key(&temp.path().join("addons").join("..").join("addons"));
    assert_eq!(direct, indirect);
    assert_eq!(direct, with_dotdot);
  }

  #[test]
  fn missing_directory_still_gets_a_stable_key() {
    let temp = tempfile::tempdir().unwrap();
    let profile = temp.path().join("not-created-yet");
    assert_eq!(profile_key(&profile), profile_key(&profile));
  }

  #[test]
  fn concurrent_holders_never_overlap() {
    let temp = tempfile::tempdir().unwrap();
    let profile = temp.path().join("addons");
    fs::create_dir_all(&profile).unwrap();

    let inside = Arc::new(AtomicUsize::new(0));
    let overlaps = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..8)
      .map(|_| {
        let profile = profile.clone();
        let inside = Arc::clone(&inside);
        let overlaps = Arc::clone(&overlaps);
        std::thread::spawn(move || {
          for _ in 0..200 {
            let _guard = lock_profile(&profile).unwrap();
            if inside.fetch_add(1, Ordering::SeqCst) != 0 {
              overlaps.fetch_add(1, Ordering::SeqCst);
            }
            std::thread::yield_now();
            inside.fetch_sub(1, Ordering::SeqCst);
          }
        })
      })
      .collect();

    for handle in handles {
      handle.join().unwrap();
    }
    assert_eq!(overlaps.load(Ordering::SeqCst), 0);
  }

  /// A nested acquire is a lost update waiting to happen, so it must fail
  /// loudly rather than deadlock or proceed unserialized.
  #[test]
  fn nested_acquire_on_one_thread_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let profile = temp.path().join("addons");
    fs::create_dir_all(&profile).unwrap();

    let outer = lock_profile(&profile).unwrap();
    let nested = lock_profile(&profile);
    assert!(
      nested.is_err(),
      "a nested profile write lock must not be granted"
    );
    drop(nested);

    // The outer hold is unaffected by the rejected attempt.
    let blocked = Arc::new(AtomicUsize::new(0));
    let handle = {
      let profile = profile.clone();
      let blocked = Arc::clone(&blocked);
      std::thread::spawn(move || {
        let _guard = lock_profile(&profile).unwrap();
        blocked.fetch_add(1, Ordering::SeqCst);
      })
    };
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert_eq!(blocked.load(Ordering::SeqCst), 0, "outer lock was released");

    drop(outer);
    handle.join().unwrap();
    assert_eq!(blocked.load(Ordering::SeqCst), 1);

    // And once released, the same thread can take it again.
    drop(lock_profile(&profile).unwrap());
  }

  #[test]
  fn different_profiles_do_not_block_each_other() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("addons");
    let second = temp.path().join("profile_work");
    fs::create_dir_all(&first).unwrap();
    fs::create_dir_all(&second).unwrap();

    let _held = lock_profile(&first).unwrap();
    let _other = lock_profile(&second).unwrap();
  }
}
