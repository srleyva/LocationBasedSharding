#![deny(missing_docs)]
//! cell_list contains code directly related to CellList
//! This includes scoring and creation
use std::collections::BTreeMap;

use s2::cellid::CellID;

use crate::{users::User, utils::ll};

/// CellScorer is the trait for a given scorer, implementing
/// this will allow you to give a custom heuristic for scoring cells
/// such as active users, total users, or some other count
pub trait CellScorer<UserCollection> {
    /// Given a `cell_list` and collection of `users` this will score the cells
    fn score_cell_list<T: User>(&self, cell_list: CellList, users: UserCollection) -> CellList
    where
        UserCollection: Iterator<Item = T>;
}

/// UserCountScorer is the a default like provided UserCountScorer
/// It scores purely on user count
pub struct UserCountScorer;

impl<UserCollection> CellScorer<UserCollection> for UserCountScorer {
    fn score_cell_list<T>(&self, mut cell_list: CellList, users: UserCollection) -> CellList
    where
        UserCollection: Iterator<Item = T>,
        T: User,
    {
        for user in users {
            let cell_id = CellID::from(user.location()).parent(cell_list.storage_level);
            let score = cell_list.cell_list.get_mut(&cell_id).unwrap();
            *score += 1;
        }
        cell_list
    }
}

/// CellList is a given order map where the key is the CellID
/// and the value is the cell score
pub struct CellList {
    storage_level: u64,
    cell_list: BTreeMap<CellID, i32>,
}

impl CellList {
    /// Generates a Collection of cells based off of the given storage level
    pub fn new(storage_level: u64) -> Self {
        let starting_cell_id = CellID::from(ll!(0.00000000, 0.00000000));
        let cell_list = Self::gather_cells(storage_level, starting_cell_id.parent(storage_level));
        Self {
            storage_level,
            cell_list,
        }
    }

    /// returns an exclusive reference to the internal cell_list
    pub fn mut_cell_list(&mut self) -> &mut BTreeMap<CellID, i32> {
        &mut self.cell_list
    }

    /// return a read reference to the cell_list
    pub fn cell_list(&self) -> &BTreeMap<CellID, i32> {
        &self.cell_list
    }

    fn gather_cells(storage_level: u64, starting_cell_id: CellID) -> BTreeMap<CellID, i32> {
        let mut seen = BTreeMap::new();
        let mut current_stack = vec![starting_cell_id];
        while let Some(current_neighbor) = current_stack.pop() {
            if !seen.contains_key(&current_neighbor) {
                current_stack.append(&mut current_neighbor.all_neighbors(storage_level));
                seen.insert(current_neighbor, 0);
            }
        }
        seen
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
