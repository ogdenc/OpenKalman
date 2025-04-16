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
 * \brief Definition of \ref not_equal_to.
 */

#ifndef OPENKALMAN_NOT_EQUAL_TO_HPP
#define OPENKALMAN_NOT_EQUAL_TO_HPP

#include <utility>
#include "basics/global-definitions.hpp"
#include "basics/language-features.hpp"

namespace OpenKalman
{
  /**
   * \brief A generalization of std::not_equal_to in which the arguments may be of different types.
   */
  template<typename T = void>
  struct not_equal_to
  {
#if defined(__cpp_static_call_operator) and __cplusplus >= 202002L
    static constexpr bool operator()(const T& lhs, const T& rhs)
#else
    constexpr bool operator()(const T& lhs, const T& rhs) const
#endif
    {
      return lhs != rhs;
    }
  };


  /**
   * \overload
   */
  template<>
  struct not_equal_to<void>
  {
    template<typename Lhs, typename Rhs>
#if defined(__cpp_static_call_operator) and __cplusplus >= 202002L
    static constexpr bool operator()(const Lhs& lhs, const Rhs& rhs)
#else
    constexpr bool operator()(const Lhs& lhs, const Rhs& rhs) const
#endif
    {
      using namespace std;
      if constexpr (is_arithmetic_v<Lhs> and is_arithmetic_v<Rhs>)
        return cmp_not_equal(lhs, rhs);
      else
        return lhs != rhs;
    }
  };


} // namespace OpenKalman

#endif //OPENKALMAN_NOT_EQUAL_TO_HPP
