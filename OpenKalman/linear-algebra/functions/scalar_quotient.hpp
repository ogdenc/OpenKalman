/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief The scalar_quotient function.
 */

#ifndef OPENKALMAN_SCALAR_QUOTIENT_HPP
#define OPENKALMAN_SCALAR_QUOTIENT_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg, typename S>
    static constexpr auto
    scalar_quotient_impl(Arg&& arg, S&& s)
    {
      if constexpr (interface::scalar_quotient_defined_for<Arg, Arg&&, S&&>)
      {
        return interface::library_interface<std::decay_t<Arg>>::scalar_quotient(std::forward<Arg>(arg), std::forward<S>(s));
      }
      else
      {
        return n_ary_operation(std::divides<scalar_type_of_t<Arg>>{}, std::forward<Arg>(arg), make_constant(arg, s));
      }
    }
  } // namespace detail


  /**
   * \brief Divide an object by a scalar value.
   */
#ifdef __cpp_concepts
  template<indexible Arg, values::scalar S> requires
    requires(S s) { {values::to_number(s)} -> std::convertible_to<scalar_type_of_t<Arg>>; }
  static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
  template<typename Arg, typename S>
  static constexpr auto
#endif
  scalar_quotient(Arg&& arg, S&& s)
  {
    if constexpr (zero<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return make_constant(std::forward<Arg>(arg),
        values::operation(
          std::divides<scalar_type_of_t<Arg>>{},
          constant_coefficient{arg},
          std::forward<S>(s)));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return to_diagonal(make_constant(diagonal_of(std::forward<Arg>(arg)),
        values::operation(
          std::divides<scalar_type_of_t<Arg>>{},
          constant_diagonal_coefficient{arg},
          std::forward<S>(s))));
    }
    else if constexpr (values::fixed_number_compares_with<S, 1>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return detail::scalar_quotient_impl(std::forward<Arg>(arg), std::forward<S>(s));
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_SCALAR_QUOTIENT_HPP
