# License Plate Segmentation and Classification

**Author:** Xander Palermo  
**Class:** CSC537 - Deep Learning  
**Instructor:** Mukulika Ghosh
**Last Accessed:** 6 May 2026

This project focuses on completing a full pipeline that processes an image of a car, and identifies and classifies
characters on its license plate into their ASCII representation. This technology has wide applications in traffic regulation,
namely administering toll fees, administering speeding tickets, and surveillance.

----

## Approach

This problem can be broken down into 3 phases:
1. License Plate Segmentation
2. Character Segmentation
3. Character Classification

To tackle each problem, a model was designed and iterated using different architectures for each phase. They are separated
into their own respective directories and designed to be interchangeable in the final pipeline implementation.
