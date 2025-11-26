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
 * \brief The scalar_product function.
 */

#ifndef OPENKALMAN_SCALAR_PRODUCT_HPP
#define OPENKALMAN_SCALAR_PRODUCT_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg, typename S>
    static constexpr auto
    scalar_product_impl(Arg&& arg, S&& s)
    {
      if constexpr (interface::scalar_product_defined_for<Arg, Arg&&, S&&>)
      {
        return interface::library_interface<stdex::remove_cvref_t<Arg>>::scalar_product(std::forward<Arg>(arg), std::forward<S>(s));
      }
      else
      {
        return n_ary_operation(std::multiplies<scalar_type_of_t<Arg>>{}, std::forward<Arg>(arg), make_constant(arg, s));
      }
    }
  }


  /**
   * \brief Multiply an object by a scalar value.
   */
#ifdef __cpp_concepts
  template<indexible Arg, values::scalar S> requires
    requires(S s) { {values::to_value_type(s)} -> std::convertible_to<scalar_type_of_t<Arg>>; }
  static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
  template<typename Arg, typename S>
  static constexpr auto
#endif
  scalar_product(Arg&& arg, S&& s)
  {
    if constexpr (zero<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return make_constant(std::forward<Arg>(arg),
        values::operation(
          std::multiplies<scalar_type_of_t<Arg>>{},
          constant_value{arg},
          std::forward<S>(s)));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return to_diagonal(make_constant(diagonal_of(std::forward<Arg>(arg)),
        values::operation(
          std::multiplies<scalar_type_of_t<Arg>>{},
          constant_diagonal_value{arg},
          std::forward<S>(s))));
    }
    else if constexpr (values::fixed_value_compares_with<S, 0>)
    {
      return make_zero(arg);
    }
    else if constexpr (values::fixed_value_compares_with<S, 1>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return detail::scalar_product_impl(std::forward<Arg>(arg), std::forward<S>(s));
    }
  }

}


#endif
