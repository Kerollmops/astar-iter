use indexmap::IndexMap;
use indexmap::map::Entry::{Occupied, Vacant};
use num_traits::Zero;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::Hash;
use std::ops::Sub;
use std::mem;

/// Compute the absolute difference between two values.
///
/// # Example
///
/// The absolute difference between 4 and 17 as unsigned values will be 13.
///
/// ```
/// use astar_iter::absdiff;
///
/// assert_eq!(absdiff(4u32, 17u32), 13u32);
/// assert_eq!(absdiff(17u32, 4u32), 13u32);
/// ```
#[inline]
pub fn absdiff<T>(x: T, y: T) -> T
where
    T: Sub<Output = T> + PartialOrd,
{
    if x < y {
        y - x
    } else {
        x - y
    }
}

pub struct AstarBagIter<N, C, FN, FH, FS> {
    to_see: BinaryHeap<SmallestCostHolder<C>>,
    min_cost: Option<C>,
    sinks: HashSet<usize>,
    parents: IndexMap<N, (HashSet<usize>, C)>,
    successors: FN,
    heuristic: FH,
    success: FS,
}

impl<N, C, FN, IN, FH, FS> AstarBagIter<N, C, FN, FH, FS>
where
    N: Eq + Hash + Clone,
    C: Zero + Ord + Copy,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = (N, C)>,
    FH: FnMut(&N) -> C,
    FS: FnMut(&N) -> bool,
{
    pub fn new(start: N, successors: FN, mut heuristic: FH, success: FS) -> Self {
        let mut to_see = BinaryHeap::new();
        let mut parents = IndexMap::new();

        to_see.push(SmallestCostHolder {
            estimated_cost: heuristic(&start),
            cost: Zero::zero(),
            index: 0,
        });

        parents.insert(start, (HashSet::new(), Zero::zero()));

        AstarBagIter {
            to_see,
            min_cost: None,
            sinks: HashSet::new(),
            parents,
            successors,
            heuristic,
            success,
        }
    }
}

impl<N, C, FN, IN, FH, FS> Iterator for AstarBagIter<N, C, FN, FH, FS>
where
    N: Eq + Hash + Clone,
    C: Zero + Ord + Copy,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = (N, C)>,
    FH: FnMut(&N) -> C,
    FS: FnMut(&N) -> bool,
{
    type Item = (AstarSolution<N>, C);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(cost_holder) = self.to_see.pop() {
            let SmallestCostHolder { cost, index, estimated_cost, .. } = cost_holder;
            if let Some(min_cost) = self.min_cost {
                if estimated_cost > min_cost {
                    // If we find that this is not the smallest node we could find we must
                    // put it back for other iterations, as we hit the _second_ min cost here.
                    self.to_see.push(cost_holder);
                    break;
                }
            }

            let successors = {
                let (node, &(_, c)) = self.parents.get_index(index).unwrap();
                if (self.success)(node) {
                    self.min_cost = Some(cost);
                    self.sinks.insert(index);
                }
                // We may have inserted a node several time into the binary heap if we found
                // a better way to access it. Ensure that we are currently dealing with the
                // best path and discard the others.
                if cost > c {
                    continue;
                }
                (self.successors)(node)
            };

            for (successor, move_cost) in successors {
                let new_cost = cost + move_cost;
                let h; // heuristic(&successor)
                let n; // index for successor
                match self.parents.entry(successor) {
                    Vacant(e) => {
                        h = (self.heuristic)(e.key());
                        n = e.index();
                        let mut p = HashSet::new();
                        p.insert(index);
                        e.insert((p, new_cost));
                    }
                    Occupied(mut e) => {
                        if e.get().1 > new_cost {
                            h = (self.heuristic)(e.key());
                            n = e.index();
                            let s = e.get_mut();
                            s.0.clear();
                            s.0.insert(index);
                            s.1 = new_cost;
                        } else {
                            if e.get().1 == new_cost {
                                // New parent with an identical cost, this is not
                                // considered as an insertion.
                                e.get_mut().0.insert(index);
                            }
                            continue;
                        }
                    }
                }

                self.to_see.push(SmallestCostHolder {
                    estimated_cost: new_cost + h,
                    cost: new_cost,
                    index: n,
                });
            }
        }

        // We must replace the current min cost by the second min cost.
        let second_min_cost = self.to_see.peek().map(|sch| sch.estimated_cost);
        let min_cost = mem::replace(&mut self.min_cost, second_min_cost);

        match min_cost {
            Some(cost) => {
                let parents = self.parents
                    .iter()
                    .map(|(k, (ps, _))| (k.clone(), ps.iter().cloned().collect()))
                    .collect();

                let solution = AstarSolution {
                    sinks: self.sinks.iter().cloned().collect(),
                    parents,
                    current: vec![],
                    terminated: false,
                };

                Some((solution, cost))
            },
            None => None,
        }
    }
}

struct SmallestCostHolder<C> {
    estimated_cost: C,
    cost: C,
    index: usize,
}

impl<C: PartialEq> PartialEq for SmallestCostHolder<C> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<C: PartialEq> Eq for SmallestCostHolder<C> {}

impl<C: Ord> PartialOrd for SmallestCostHolder<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Ord> Ord for SmallestCostHolder<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.cmp(&other.cost),
            s => s,
        }
    }
}

#[derive(Clone)]
pub struct AstarSolution<N> {
    sinks: Vec<usize>,
    parents: Vec<(N, Vec<usize>)>,
    current: Vec<Vec<usize>>,
    terminated: bool,
}

impl<N: Clone + Eq + Hash> AstarSolution<N> {
    fn complete(&mut self) {
        loop {
            let ps = match self.current.last() {
                None => self.sinks.clone(),
                Some(last) => {
                    let &top = last.last().unwrap();
                    self.parents(top).clone()
                }
            };
            if ps.is_empty() {
                break;
            }
            self.current.push(ps);
        }
    }

    fn next_vec(&mut self) {
        while self.current.last().map(Vec::len) == Some(1) {
            self.current.pop();
        }
        self.current.last_mut().map(Vec::pop);
    }

    fn node(&self, i: usize) -> &N {
        &self.parents[i].0
    }

    fn parents(&self, i: usize) -> &Vec<usize> {
        &self.parents[i].1
    }
}

impl<N: Clone + Eq + Hash> Iterator for AstarSolution<N> {
    type Item = Vec<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.terminated {
            return None;
        }
        self.complete();
        let path = self
            .current
            .iter()
            .rev()
            .map(|v| v.last().cloned().unwrap())
            .map(|i| self.node(i).clone())
            .collect::<Vec<_>>();
        self.next_vec();
        self.terminated = self.current.is_empty();
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easy() {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        struct Pos(i32, i32);

        impl Pos {
            fn distance(&self, other: &Pos) -> u32 {
                (absdiff(self.0, other.0) + absdiff(self.1, other.1)) as u32
            }

            fn successors(&self) -> Vec<(Pos, u32)> {
                let &Pos(x, y) = self;
                vec![
                    Pos(x+1,y+2), Pos(x+1,y-2), Pos(x-1,y+2), Pos(x-1,y-2),
                    Pos(x+2,y+1), Pos(x+2,y-1), Pos(x-2,y+1), Pos(x-2,y-1),
                ].into_iter().map(|p| (p, 1)).collect()
            }
        }

        static GOAL: Pos = Pos(4, 6);

        let mut astar_iter = AstarBagIter::new(
            Pos(1, 1), // start
            |p| p.successors(), // successors
            |p| p.distance(&GOAL) / 3, // heuristic
            |p| *p == GOAL, // success
        );

        assert_eq!(astar_iter.next().expect("no path found").1, 4);
        assert_eq!(astar_iter.next().expect("no path found").1, 5);
        assert_eq!(astar_iter.next().expect("no path found").1, 6);
        assert_eq!(astar_iter.next().expect("no path found").1, 7);
        assert_eq!(astar_iter.next().expect("no path found").1, 8);
        assert_eq!(astar_iter.next().expect("no path found").1, 9);
        assert_eq!(astar_iter.next().expect("no path found").1, 10);
        // and so far so on...
    }
}
