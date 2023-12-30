//! Mostly various extension traits for iterators, not too dissimilar
//! from itertools. Optimised for personal use, but potentially
//! generally useful.

#![cfg_attr(not(feature = "std"), no_std)]

use core::{
    cmp::Ordering,
    fmt::{self, Debug, Formatter},
    mem::{swap, zeroed},
    ops::{Add, Div, Index, Range, Sub},
};

/// Shared error type.
#[derive(PartialEq, Eq, Debug)]
pub enum PreludeError {
    /// A data structure is full.
    Full,
}

/// A stack-allocated ring buffer.
///
/// The default mode is FIFO, but it can be used as a LIFO stack as well.
#[derive(Copy, Clone, Hash)]
pub struct RingBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    start: usize,
    end: usize,
    full: bool,
}

impl<T, const N: usize> RingBuffer<T, N> {
    /// Creates a new ring buffer, filling it with `elem`.
    pub fn new() -> Self {
        let buffer = unsafe { zeroed() };
        Self {
            buffer,
            start: 0,
            end: 0,
            full: false,
        }
    }

    /// Pushes `elem` into the buffer. Returns an error if the buffer
    /// is full.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 3>::new();
    /// assert_eq!(buffer.try_push(1), Ok(()));
    /// assert_eq!(buffer.try_push(2), Ok(()));
    /// assert_eq!(buffer.try_push(3), Ok(()));
    /// assert_eq!(buffer.try_push(4), Err(PreludeError::Full));
    /// ```
    pub fn try_push(&mut self, elem: T) -> Result<(), PreludeError> {
        if self.is_full() {
            return Err(PreludeError::Full);
        }
        self.push(elem);
        Ok(())
    }

    /// Pushes `elem` into the buffer, overwriting the oldest element
    /// if the buffer is full.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 3>::new();
    /// buffer.push(1);
    /// buffer.push(2);
    /// buffer.push(3);
    /// buffer.push(4);
    /// assert_eq!(buffer.pop(), Some(2));
    /// assert_eq!(buffer.pop(), Some(3));
    /// assert_eq!(buffer.pop(), Some(4));
    /// ```
    pub fn push(&mut self, elem: T) {
        if self.is_full() {
            self.start = (self.start + 1) % N;
        }
        self.buffer[self.end] = Some(elem);
        self.end = (self.end + 1) % N;
        if self.end == self.start {
            self.full = true;
        }
    }

    /// Pops an element from the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::from([1u32, 2, 3]);
    /// assert_eq!(buffer.pop(), Some(1));
    /// assert_eq!(buffer.pop(), Some(2));
    /// assert_eq!(buffer.pop(), Some(3));
    /// assert_eq!(buffer.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return None;
        }
        self.full = false;
        let elem = self.buffer[self.start].take();
        self.start = (self.start + 1) % N;
        elem
    }

    /// Pops an element from the back of the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::from([1u32, 2, 3]);
    /// assert_eq!(buffer.pop_back(), Some(3));
    /// assert_eq!(buffer.pop_back(), Some(2));
    /// assert_eq!(buffer.pop_back(), Some(1));
    /// assert_eq!(buffer.pop_back(), None);
    /// ```
    pub fn pop_back(&mut self) -> Option<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return None;
        }
        self.full = false;
        self.end = (self.end + N - 1) % N;
        self.buffer[self.end].take()
    }

    /// Returns the element at `i` in the ring buffer, as
    /// counted from the start of the buffer, i.e. the Nth oldest
    /// element.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::from([1u32, 2, 3]);
    /// assert_eq!(buffer.get(0), Some(1));
    /// assert_eq!(buffer.get(1), Some(2));
    /// assert_eq!(buffer.get(2), Some(3));
    /// assert_eq!(buffer.get(3), None);
    /// ```
    pub fn get(&self, i: usize) -> Option<T>
    where
        T: Clone,
    {
        if i >= self.len() {
            return None;
        }
        let index = (self.start + i) % N;
        self.buffer[index].clone()
    }

    /// Returns the number of entries in the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 5>::new();
    /// buffer.extend([1, 2, 3]);
    /// assert_eq!(buffer.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        if self.full {
            N
        } else if self.end >= self.start {
            self.end - self.start
        } else {
            N - self.start + self.end
        }
    }

    /// Returns `true` if the ring buffer is empty.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 3>::new();
    /// assert!(buffer.is_empty());
    /// buffer.push(1);
    /// assert!(!buffer.is_empty());
    /// buffer.pop();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.end == self.start && !self.is_full()
    }

    /// Returns `true` if the ring buffer is full.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 1>::new();
    /// assert!(!buffer.is_full());
    /// buffer.push(1);
    /// assert!(buffer.is_full());
    /// buffer.pop();
    /// assert!(!buffer.is_full());
    /// ```
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Returns the maximum capacity of the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 5>::new();
    /// assert_eq!(buffer.capacity(), 5);
    /// ```
    pub fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of empty spaces in the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::<u32, 5>::new();
    /// assert_eq!(buffer.space(), 5);
    /// buffer.push(1);
    /// assert_eq!(buffer.space(), 4);
    /// ```
    pub fn space(&self) -> usize {
        N - self.len()
    }

    /// Clears the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let mut buffer = RingBuffer::from([1u32, 2, 3]);
    /// buffer.clear();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.start = 0;
        self.end = 0;
        self.full = false;
        self.buffer.iter_mut().for_each(|slot| *slot = None);
    }

    /// Returns an iterator over the ring buffer.
    ///
    /// ```
    /// # use prelude::*;
    /// let buffer = RingBuffer::from([1u32, 2, 3]);
    /// let mut iter = buffer.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> RingBufferIter<'_, T, N> {
        RingBufferIter {
            buffer: self,
            index: 0,
            index_back: self.len(),
        }
    }
}

impl<T, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Debug for RingBuffer<T, N>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Ring buffer: ")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> IntoIterator for RingBuffer<T, N>
where
    T: Clone,
{
    type Item = T;
    type IntoIter = RingBufferIntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        RingBufferIntoIter {
            buffer: self,
            index: 0,
            index_back: len,
        }
    }
}

/// An iterator over a ring buffer, created by [`RingBuffer::into_iter`].
pub struct RingBufferIntoIter<T, const N: usize> {
    buffer: RingBuffer<T, N>,
    index: usize,
    index_back: usize,
}

impl<T, const N: usize> Iterator for RingBufferIntoIter<T, N>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.index_back {
            return None;
        }
        let index = (self.buffer.start + self.index) % N;
        self.index += 1;
        self.buffer.buffer[index].clone()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<T, const N: usize> DoubleEndedIterator for RingBufferIntoIter<T, N>
where
    T: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index_back == self.index {
            return None;
        }
        self.index_back -= 1;
        let index = (self.buffer.start + self.index_back) % N;
        self.buffer.buffer[index].clone()
    }
}

impl<T, const N: usize> ExactSizeIterator for RingBufferIntoIter<T, N>
where
    T: Clone,
{
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// An iterator over a ring buffer, created by [`RingBuffer::iter`].
pub struct RingBufferIter<'a, T, const N: usize> {
    buffer: &'a RingBuffer<T, N>,
    index: usize,
    index_back: usize,
}

impl<'a, T, const N: usize> Iterator for RingBufferIter<'a, T, N>
where
    T: Clone,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.index_back {
            return None;
        }
        let index = (self.buffer.start + self.index) % N;
        self.index += 1;
        self.buffer.buffer[index].as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for RingBufferIter<'a, T, N>
where
    T: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index_back == self.index {
            return None;
        }
        self.index_back -= 1;
        let index = (self.buffer.start + self.index_back) % N;
        self.buffer.buffer[index].as_ref()
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for RingBufferIter<'a, T, N>
where
    T: Clone,
{
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<T, const N: usize> From<[T; N]> for RingBuffer<T, N> {
    fn from(array: [T; N]) -> Self {
        Self {
            buffer: array.map(Some),
            start: 0,
            end: N,
            full: true,
        }
    }
}

impl<T, const N: usize> FromIterator<T> for RingBuffer<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::new();
        for item in iter {
            buffer
                .try_push(item)
                .expect("RingBuffer::from_iter: buffer full");
        }
        buffer
    }
}

impl<T, const N: usize> Extend<T> for RingBuffer<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T, const N: usize> Index<usize> for RingBuffer<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("RingBuffer::index: index out of bounds");
        }
        let index = (self.start + index) % N;
        self.buffer[index].as_ref().unwrap()
    }
}

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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        (lower.saturating_sub(1), upper.map(|u| u.saturating_sub(1)))
    }
}

impl<I> ExactSizeIterator for Derivative<I>
where
    I: Iterator + ExactSizeIterator,
    I::Item: Copy + Sub,
{
    fn len(&self) -> usize {
        self.iter.len().saturating_sub(1)
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

    #[test]
    fn ring_buffer_pop_empty() {
        let mut buffer = RingBuffer::<u32, 3>::new();
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn ring_buffer_pop_not_full() {
        let mut buffer = RingBuffer::<u32, 3>::new();
        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        assert_eq!(buffer.pop(), Some(1));
    }

    #[test]
    fn ring_buffer_pop_back_back_empty() {
        let mut buffer = RingBuffer::<u32, 3>::new();
        assert_eq!(buffer.pop_back(), None);
    }

    #[test]
    fn ring_buffer_pop_back_not_full() {
        let mut buffer = RingBuffer::<u32, 3>::new();
        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        assert_eq!(buffer.pop_back(), Some(2));
    }

    #[test]
    fn ring_buffer_is_empty_when_full() {
        let buffer = RingBuffer::from([1u32, 2, 3]);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn ring_buffer_from_slice_is_full() {
        let buffer = RingBuffer::from([1u32, 2, 3]);
        assert!(buffer.is_full());
    }

    #[test]
    fn ring_buffer_iter_rev() {
        let buffer = RingBuffer::from([1u32, 2, 3]);
        let mut iter = buffer.iter().rev();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn ring_buffer_iter_from_both_ends() {
        let buffer = RingBuffer::from([1u32, 2, 3]);
        let mut iter = buffer.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
    }
}
