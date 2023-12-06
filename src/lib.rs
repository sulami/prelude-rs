#![no_std]

use core::{
    cmp::Ordering,
    ops::{Add, Div, Range, Sub},
};

/// An iterator that returns items from the wrapped iterator in chunks
/// of `N`. If the last chunk cannot be filled completely, it is
/// filled up with [`Default::default`].
///
/// After the last chunk, calls to [`Chunks::next`] return [`None`].
///
/// ```
/// # use prelude::*;
/// let mut iter = (0u8..10).chunks::<3>();
/// assert_eq!(iter.next(), Some([0,1,2]));
/// assert_eq!(iter.next(), Some([3,4,5]));
/// assert_eq!(iter.next(), Some([6,7,8]));
/// assert_eq!(iter.next(), Some([9,0,0]));
/// assert_eq!(iter.next(), None);
/// ```
pub struct Chunks<I, const N: usize>
where
    I: Iterator,
    I::Item: Default + Copy,
{
    iter: I,
}

impl<I, const N: usize> Iterator for Chunks<I, N>
where
    I: Iterator,
    I::Item: Default + Copy,
{
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        let mut rv = [I::Item::default(); N];
        for (i, slot) in rv.iter_mut().enumerate() {
            if let Some(item) = self.iter.next() {
                *slot = item;
            } else if i == 0 {
                return None;
            }
        }
        Some(rv)
    }
}

pub trait Chunkable {
    fn chunks<const N: usize>(self) -> Chunks<Self, N>
    where
        Self: Iterator + Sized,
        Self::Item: Default + Copy;
}

impl<I> Chunkable for I
where
    I: Iterator,
    I::Item: Default + Copy,
{
    fn chunks<const N: usize>(self) -> Chunks<Self, N> {
        Chunks { iter: self }
    }
}

/// Use binary search to search an arbitrary space by suppling an
/// input range and a function that returns an ordering relative to
/// the target value. Returns the matching number from the range.
///
/// The [`Range`] can use any integer type except [`i8`].
///
/// The simplest example:
/// ```
/// # use prelude::*;
/// assert_eq!(binary_search_range(0_u32..10, |i| { i.cmp(&3) }), 3)
/// ```
///
/// One can reimplement [`slice::binary_search`]:
/// ```
/// # use prelude::*;
/// let arr = [4u32, 8, 15, 16, 23, 42];
/// assert_eq!(binary_search_range(0_usize..arr.len(), |i| { arr[i].cmp(&23) }), 4)
/// ```
pub fn binary_search_range<T, F>(range: Range<T>, f: F) -> T
where
    T: Copy + Ord + From<u8> + Add<Output = T> + Div<Output = T> + Sub<Output = T>,
    F: Fn(T) -> Ordering,
{
    let mut lower = range.start;
    let mut upper = range.end;

    while lower < upper {
        let mid = lower + (upper - lower) / 2.into();
        match f(mid) {
            Ordering::Greater => {
                upper = mid;
            }
            Ordering::Equal => {
                break;
            }
            Ordering::Less => {
                lower = mid + 1.into();
            }
        }
    }

    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_range_works_with_various_number_types() {
        assert_eq!(binary_search_range(0_u8..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_u16..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_u32..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_u64..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_u128..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_usize..10, |i| { i.cmp(&3) }), 3);
        // i8 doesn't work because one can't convert it to u8.
        assert_eq!(binary_search_range(0_i16..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_i32..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_i64..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_i128..10, |i| { i.cmp(&3) }), 3);
        assert_eq!(binary_search_range(0_isize..10, |i| { i.cmp(&3) }), 3);
    }
}