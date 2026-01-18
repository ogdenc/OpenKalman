/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_identity_matrix.
 */

#ifndef OPENKALMAN_MAKE_IDENTITY_MATRIX_HPP
#define OPENKALMAN_MAKE_IDENTITY_MATRIX_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/identity_matrix.hpp"
#include "make_constant_diagonal.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref identity_matrix with a given shape pattern.
   * \tparam The type of the value.
   * \param p a \ref patterns::pattern_collection for the result (e.g., an std::extents object).
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P>
  constexpr identity_matrix auto
#else
  template<typename C, typename P, std::enable_if_t<values::value<C> and patterns::pattern_collection<P>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix(P&& p)
  {
    values::fixed_value<values::value_type_of_t<C>, 1> c;
    return make_constant_diagonal(std::move(c), std::forward<P>(p));
  }


  /**
   * \overload
   * \brief The \ref patterns::pattern_collection "pattern_collection" is constructed from a list of \ref patterns::patterns "patterns".
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern...Ps>
  constexpr identity_matrix auto
#else
  template<typename C, typename...Ps, std::enable_if_t<values::value<C> and (... and patterns::pattern<Ps>), int> = 0>
  constexpr auto
#endif
  make_identity_matrix(Ps&&...ps)
  {
    return make_identity_matrix<C>(std::tuple{std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_diagonal_object based on a default-initializable \ref patterns::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr identity_matrix auto
#else
  template<typename C, typename P, std::enable_if_t<
    values::value<C> and
    patterns::pattern_collection<P> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix()
  {
    return make_identity_matrix<C>(P{});
  }


}

#endif
