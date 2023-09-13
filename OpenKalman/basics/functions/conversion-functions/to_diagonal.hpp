/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_HPP
#define OPENKALMAN_TO_DIAGONAL_HPP

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename = void>
    struct to_diagonal_exists : std::false_type {};

    template<typename Arg>
    struct to_diagonal_exists<Arg, std::void_t<decltype(
      interface::LibraryRoutines<std::decay_t<Arg>>::template to_diagonal(std::declval<Arg&&>()))>> : std::true_type {};
  } // namespace detail
#endif


  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<vector<0, Likelihood::maybe> Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, std::enable_if_t<vector<Arg, 0, Likelihood::maybe>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg)
  {
    using Interface = interface::LibraryRoutines<std::decay_t<Arg>>;

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
#ifdef __cpp_concepts
    else if constexpr (requires { Interface::template to_diagonal(std::forward<Arg>(arg)); })
#else
    else if constexpr (detail::to_diagonal_exists<Arg>::value)
#endif
    {
      return Interface::to_diagonal(std::forward<Arg>(arg));
    }
    else
    {
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_DIAGONAL_HPP
