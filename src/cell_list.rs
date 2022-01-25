#![deny(missing_docs)]
//! cell_list contains code directly related to CellList
//! This includes scoring and creation
use std::collections::BTreeMap;

use s2::cellid::CellID;

use crate::{users::UserCollection, utils::ll};

/// CellScorer is the trait for a given scorer, implementing
/// this will allow you to give a custom heuristic for scoring cells
/// such as active users, total users, or some other count
pub trait CellScorer {
    /// Given a `cell_list` and collection of `users` this will score the cells
    fn score_cell_list(&self, cell_list: CellList, users: UserCollection) -> CellList;
}

/// UserCountScorer is the a default like provided UserCountScorer
/// It scores purely on user count
pub struct UserCountScorer;

impl CellScorer for UserCountScorer {
    fn score_cell_list(&self, mut cell_list: CellList, users: UserCollection) -> CellList {
        for user in users {
            let cell_id = CellID::from(user.location()).parent(cell_list.storage_level);
            let score = cell_list.cell_list.get_mut(&cell_id).unwrap();
            *score += 1;
        }
        cell_list
    }
}

pub struct CellList {
    storage_level: u64,
    cell_list: BTreeMap<CellID, i32>,
}

impl CellList {
    pub fn new(storage_level: u64) -> Self {
        let starting_cell_id = CellID::from(ll!(0.00000000, 0.00000000));
        let mut cell_list = BTreeMap::new();
        let child_tread = Builder::new()
            .stack_size(50 * 1024 * 1024)
            .spawn(move || {
                Self::gather_cells(storage_level, starting_cell_id, &mut cell_list);
                cell_list
            })
            .expect("could not spawn thread to build s2 cells");

        let cell_list = child_tread.join().expect("err building s2 list");

        Self {
            storage_level,
            cell_list,
        }
    }

    pub fn mut_cell_list(&mut self) -> &mut BTreeMap<CellID, i32> {
        &mut self.cell_list
    }

    pub fn scored_cells(&self) -> &BTreeMap<CellID, i32> {
        &self.cell_list
    }

    fn gather_cells(storage_level: u64, cell_id: CellID, seen: &mut BTreeMap<CellID, i32>) {
        let current_cell_neighbors = cell_id.vertex_neighbors(storage_level);
        for neighbor in current_cell_neighbors {
            match seen.get(&neighbor) {
                Some(_) => (),
                None => {
                    seen.insert(neighbor, 0);
                    Self::gather_cells(storage_level, neighbor, seen);
                }
            };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_geoshard_cell_list() {
        let cell_list = CellList::new(8).cell_list;
        assert_eq!(cell_list.len(), 393216);
    }
}
