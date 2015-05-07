---
layout: post
title: Coding and Documentation Style
category : community
tagline:
tags : [coding, contribution, style]
---
{% include JB/setup %}

## Coding Styling
We follow Google's C++ coding [style](http://google-styleguide.googlecode.com/svn/trunk/cppguide.html)

## Documentation Style

### Comments

We follow the [Doxygen](http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html)
style to add comments for the code. There are at least three points we need to follow,

1. **JAVADOC_AUTOREF** is enabled to write the comment block as,

        /**
         * brief description with a dot at the end.
         * detailed description.
         * @param
         * @param
         * @return
         */

2. One line comment should be like this (actually there are more choices. We use this one),

        int num_procs; /**> total number of mpi processes */

    the comments after double slashes `//` will not be generated to the html API pages but are allowed.

3. Comments are apart from code should use macros like,

        /**
         * \file this file, solver.h, contains two classes \class Solver and \class Prefetcher
         */

### Tutorial and User Guide
