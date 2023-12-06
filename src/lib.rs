#![no_std]

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
pub struct Chunks<I: Iterator, const N: usize>
where
    I::Item: Default + Copy,
{
    iter: I,
}

impl<I: Iterator, const N: usize> core::iter::Iterator for Chunks<I, N>
where
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


#[cfg(test)]
mod tests {
    use super::*;
}
