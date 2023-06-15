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
 * \brief Definitions for \ref make_triangular_matrix.
 */

#ifndef OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP
#define OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, TriangleType t, typename = void>
    struct make_triangular_matrix_defined: std::false_type {};

    template<typename T, TriangleType t>
    struct make_triangular_matrix_defined<T, t, std::void_t<
      decltype(interface::TriangularTraits<std::decay_t<T>>::template make_triangular_matrix<t>(std::declval<T&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Create a \ref triangular_matrix from a general matrix.
   * \tparam t The intended \ref TriangleType of the result.
   * \tparam Arg A general matrix to be made triangular.
   */
#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, square_matrix<Likelihood::maybe> Arg> requires
    (t == TriangleType::lower or t == TriangleType::upper or t == TriangleType::diagonal)
  constexpr triangular_matrix<t> decltype(auto)
#else
  template<TriangleType t = TriangleType::lower, typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
    (t == TriangleType::lower or t == TriangleType::upper or t == TriangleType::diagonal), int> = 0>
  constexpr decltype(auto)
#endif
  make_triangular_matrix(Arg&& arg)
  {
    using Traits = interface::TriangularTraits<std::decay_t<Arg>>;

    if constexpr (triangular_matrix<Arg, t>)
    {
      // If Arg is already triangular of type t, pass through to the result
      return std::forward<Arg>(arg);
    }
    else if constexpr (t == TriangleType::diagonal)
    {
      return to_diagonal(diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (triangular_matrix<Arg, TriangleType::any, Likelihood::maybe> and not triangular_matrix<Arg, t, Likelihood::maybe>)
    {
      // Arg is the opposite triangle of t.
      return make_triangular_matrix<TriangleType::diagonal>(std::forward<Arg>(arg));
    }
    else if constexpr (triangular_adapter<Arg>)
    {
      // If Arg is a triangular adapter but was not known to be square at compile time, return a result guaranteed to be square triangular.
      return make_triangular_matrix<t>(nested_matrix(std::forward<Arg>(arg)));
    }
# ifdef __cpp_concepts
    else if constexpr (requires (Arg&& arg) { Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg)); })
# else
    else if constexpr (detail::make_triangular_matrix_defined<Arg, t>::value)
# endif
    {
      auto new_t {Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg))};
      static_assert(triangular_matrix<decltype(new_t), t>, "make_triangular_matrix interface must return a triangular matrix");
      return new_t;
    }
    else // Default behavior if interface function not defined:
    {
      using pArg = std::conditional_t<std::is_lvalue_reference_v<Arg>, Arg, std::remove_reference_t<decltype(make_self_contained(arg))>>;
      return TriangularMatrix<pArg, t> {std::forward<Arg>(arg)};
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_TRIANGULAR_MATRIX_HPP
