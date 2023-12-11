#![no_std]

use core::{
    cmp::Ordering,
    mem::swap,
    ops::{Add, Div, Range, Sub},
};

/// An iterator that can return its items in chunks instead of one by
/// one.
pub trait Chunkable {
    /// Return chunks of items. If the iterator runs out without
    /// filling up the last chunk, the item's [`Default::default`]
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
    fn chunks<const N: usize>(self) -> Chunks<Self, N>
    where
        Self: Iterator + Sized,
        Self::Item: Default + Copy;
}

/// An iterator that returns items from the wrapped iterator in chunks
/// of `N`. Created by [`chunks`](Chunkable::chunks).
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

/// A collection that can be homogenous or heterogenous.
pub trait Genous {
    /// Returns true if all elements in `coll` are equal.
    ///
    /// ```
    /// # use prelude::*;
    /// assert!([1, 1, 1].is_homogenous());
    /// assert!(![1, 2, 1].is_homogenous());
    /// ```
    fn is_homogenous(&self) -> bool;

    /// Returns true if the elements in `coll` are not all equal.
    ///
    /// ```
    /// # use prelude::*;
    /// assert!([1, 2, 1].is_heterogenous());
    /// assert!(![1, 1, 1].is_heterogenous());
    /// ```
    fn is_heterogenous(&self) -> bool;
}

impl<T> Genous for T
where
    T: IntoIterator + Copy,
    T::Item: PartialEq,
{
    fn is_homogenous(&self) -> bool {
        let mut iter = self.into_iter();
        if let Some(item) = iter.next() {
            for other in iter {
                if other != item {
                    return false;
                }
            }
        }
        true
    }

    fn is_heterogenous(&self) -> bool {
        let mut iter = self.into_iter();
        if let Some(item) = iter.next() {
            for other in iter {
                if other != item {
                    return true;
                }
            }
        }
        false
    }
}

/// An iterator that can be searched by inspecting two elements at a
/// time.
pub trait WindowSearchable {
    type Item;

    /// Search the iterator by looking at two items at a time. Returns
    /// the first pair that matches the predicate.
    ///
    /// ```
    /// # use prelude::*;
    /// assert_eq!([1, 2, 3, 3, 4, 5].iter().window_search(|a, b| a == b), Some((&3, &3)));
    /// ```
    fn window_search<F>(self, f: F) -> Option<(Self::Item, Self::Item)>
    where
        F: Fn(Self::Item, Self::Item) -> bool;
}

impl<I> WindowSearchable for I
where
    I: Iterator,
    I::Item: Copy,
{
    type Item = I::Item;

    fn window_search<F>(mut self, f: F) -> Option<(I::Item, I::Item)>
    where
        F: Fn(I::Item, I::Item) -> bool,
    {
        if let Some(mut item) = self.next() {
            for other in self {
                if f(item, other) {
                    return Some((item, other));
                }
                item = other;
            }
        }
        None
    }
}

/// An iterator that can produce overlapping windows of its items.
pub trait Overlappable {
    /// Return overlapping windows of items. Stops as soon as the
    /// window cannot be filled.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut iter = (0u8..=5).overlapping_windows::<3>();
    /// assert_eq!(iter.next(), Some([0, 1, 2]));
    /// assert_eq!(iter.next(), Some([1, 2, 3]));
    /// assert_eq!(iter.next(), Some([2, 3, 4]));
    /// assert_eq!(iter.next(), Some([3, 4, 5]));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn overlapping_windows<const N: usize>(self) -> OverlappingWindows<Self, N>
    where
        Self: Iterator + Sized,
        Self::Item: Copy;
}

/// An iterator that produces overlapping windows of the wrapped
/// iterator. Created by
/// [`overlapping_windows`](Overlappable::overlapping_windows).
pub struct OverlappingWindows<I, const N: usize>
where
    I: Iterator,
    I::Item: Copy,
{
    iter: I,
    finished: bool,
    head: usize,
    buf: [Option<I::Item>; N],
}

impl<I, const N: usize> Iterator for OverlappingWindows<I, N>
where
    I: Iterator,
    I::Item: Copy,
{
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.buf[0].is_none() {
            // Fresh buffer, initialise.
            for slot in self.buf.iter_mut() {
                match self.iter.next() {
                    Some(item) => {
                        *slot = Some(item);
                    }
                    None => {
                        self.finished = true;
                        return None;
                    }
                }
            }
        } else {
            // Fetch the next item.
            match self.iter.next() {
                Some(item) => {
                    self.buf[self.head] = Some(item);
                    self.head = (self.head + 1) % N;
                }
                None => {
                    self.finished = true;
                    return None;
                }
            }
        }

        let mut rv = [self.buf[0].unwrap(); N];
        self.buf
            .iter()
            .skip(self.head)
            .enumerate()
            .for_each(|(i, item)| rv[i] = item.unwrap());
        self.buf
            .iter()
            .take(self.head)
            .enumerate()
            .for_each(|(i, item)| rv[N - self.head + i] = item.unwrap());

        Some(rv)
    }
}

impl<I> Overlappable for I
where
    I: Iterator,
    I::Item: Copy,
{
    fn overlapping_windows<const N: usize>(self) -> OverlappingWindows<Self, N> {
        if N == 0 {
            panic!("window size 0 is invalid");
        }
        OverlappingWindows {
            iter: self,
            finished: false,
            head: 0,
            buf: [None; N],
        }
    }
}

/// Something that looks like an integer.
pub trait Integeresque {
    /// Least common multiple.
    fn lcm(self, other: Self) -> Self;
    /// Greatest common divisor.
    fn gcd(self, other: Self) -> Self;
}

macro_rules! impl_integeresque {
    ($t:ty) => {
        impl Integeresque for $t {
            fn lcm(self, other: Self) -> Self {
                self * other / self.gcd(other)
            }
            fn gcd(self, other: Self) -> Self {
                let mut max = self;
                let mut min = other;
                if min > max {
                    swap(&mut max, &mut min);
                }

                loop {
                    let res = max % min;
                    if res == 0 {
                        return min;
                    }

                    max = min;
                    min = res;
                }
            }
        }
    };
}

impl_integeresque!(u8);
impl_integeresque!(u16);
impl_integeresque!(u32);
impl_integeresque!(u64);
impl_integeresque!(u128);
impl_integeresque!(usize);
impl_integeresque!(i8);
impl_integeresque!(i16);
impl_integeresque!(i32);
impl_integeresque!(i64);
impl_integeresque!(i128);
impl_integeresque!(isize);

/// An iterator which elements can be derived or integrated.
pub trait Differential {
    /// Returns the difference between each two items. By definition,
    /// the resulting sequence is shorter by one than the original
    /// one.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut iter = [4, 8, 15, 16, 23, 42].iter().derivative();
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(7));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(7));
    /// assert_eq!(iter.next(), Some(19));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn derivative(self) -> Derivative<Self>
    where
        Self: Iterator + Sized,
        Self::Item: Copy + Sub;
}

/// An iterator that returns the difference between the items of the
/// wrapped iterator. Created by
/// [`derivative`](Differential::derivative).
pub struct Derivative<I>
where
    I: Iterator,
    I::Item: Copy + Sub,
{
    iter: I,
    last: Option<I::Item>,
}

impl<I> Iterator for Derivative<I>
where
    I: Iterator,
    I::Item: Copy + Sub,
{
    type Item = <I::Item as Sub>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last.is_none() {
            match self.iter.next() {
                None => return None,
                Some(item) => {
                    self.last = Some(item);
                }
            }
        }
        match self.iter.next() {
            None => None,
            Some(item) => {
                let dx = item - self.last.unwrap();
                self.last = Some(item);
                Some(dx)
            }
        }
    }
}

impl<I> Differential for I
where
    I: Iterator,
    I::Item: Copy + Sub,
{
    fn derivative(self) -> Derivative<Self> {
        Derivative {
            iter: self,
            last: None,
        }
    }
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

    #[test]
    fn lcm_works_with_various_number_types() {
        assert_eq!(3_u8.lcm(4), 12);
        assert_eq!(3_u16.lcm(4), 12);
        assert_eq!(3_u32.lcm(4), 12);
        assert_eq!(3_u64.lcm(4), 12);
        assert_eq!(3_u128.lcm(4), 12);
        assert_eq!(3_usize.lcm(4), 12);
        assert_eq!(3_i8.lcm(4), 12);
        assert_eq!(3_i16.lcm(4), 12);
        assert_eq!(3_i32.lcm(4), 12);
        assert_eq!(3_i64.lcm(4), 12);
        assert_eq!(3_i128.lcm(4), 12);
        assert_eq!(3_isize.lcm(4), 12);
    }
}
