/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of callable objects equivalent to std::ranges::equal_to, etc.
 */

#ifndef OPENKALMAN_COMPATIBILITY_RANGES_ALGORITHM_HPP
#define OPENKALMAN_COMPATIBILITY_RANGES_ALGORITHM_HPP

#include <algorithm>
#include "basics/compatibility/iterator.hpp"
#include "basics/compatibility/ranges/range-concepts.hpp"

namespace OpenKalman::stdex::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::in_out_result;
  using std::ranges::copy;
  using std::ranges::copy_result;
#else
  template<typename I, typename O>
  struct in_out_result
  {
    I in;
    O out;

    template<typename I2, typename O2, std::enable_if_t<
      stdex::convertible_to<const I&, I2> and
      stdex::convertible_to<const O&, O2>, int> = 0>
    constexpr operator
    in_out_result<I2, O2>() const &
    {
      return {in, out};
    }

    template<class I2, class O2, std::enable_if_t<
      stdex::convertible_to<I, I2> and
      stdex::convertible_to<O, O2>, int> = 0>
    constexpr operator
    in_out_result<I2, O2>() &&
    {
      return {std::move(in), std::move(out)};
    }
  };


  template<typename I, typename O>
  using copy_result = in_out_result<I, O>;


  namespace detail
  {
    struct copy_fn
    {
      template<typename I, typename S, typename O, std::enable_if_t<
        stdex::input_iterator<I> and
        stdex::sentinel_for<I, S> and
        stdex::weakly_incrementable<O> and
        stdex::indirectly_copyable<I, O>, int> = 0>
      constexpr copy_result<I, O>
      operator()(I first, S last, O result) const
      {
        for (; first != last; ++first, (void)++result)
          *result = *first;
        return {std::move(first), std::move(result)};
      }


      template<typename R, typename O, std::enable_if_t<
        input_range<R> and
        stdex::weakly_incrementable<O> and
        indirectly_copyable<iterator_t<R>, O>, int> = 0>
      constexpr copy_result<borrowed_iterator_t<R>, O>
      operator()(R&& r, O result) const
      {
        return (*this)(ranges::begin(r), ranges::end(r), std::move(result));
      }
    };
  }

  inline constexpr detail::copy_fn copy;
#endif

}

#endif
