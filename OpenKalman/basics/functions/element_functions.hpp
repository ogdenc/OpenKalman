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
 * \brief Overloaded general functions relating to elements.
 */

#ifndef OPENKALMAN_ELEMENT_FUNCTIONS_HPP
#define OPENKALMAN_ELEMENT_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // \todo Add functions that return stl-compatible iterators.

  // =================== //
  //  Element functions  //
  // =================== //

  namespace detail
  {
    template<bool set, typename Arg, std::size_t...seq, typename...I>
    constexpr void check_index_bounds(const Arg& arg, std::index_sequence<seq...>, I...i)
    {
      if constexpr (sizeof...(I) == 1)
      {
        std::size_t dim_i = (i,...);
        std::size_t c = get_index_dimension_of<1>(arg);
        if (c == 1)
        {
          std::size_t r = get_index_dimension_of<0>(arg);
          if (dim_i >= r)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Row index is " +
            std::to_string(dim_i) + " but should be in range [0..." + std::to_string(r-1) + "].")};
        }
        else
        {
          if (dim_i >= c)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Column index is " +
            std::to_string(dim_i) + " but should be range [0..." + std::to_string(c-1) + "].")};
        }
      }
      else
      {
        (((static_cast<std::size_t>(i) >= std::size_t{get_index_dimension_of<seq>(arg)}) ?
          throw std::out_of_range {(("At least one " + std::string {set ? "s" : "g"} +
            "et_element index out of range:") + ... + (" Index " + std::to_string(seq) + " is " +
            std::to_string(i) + " and should be in range [0..." +
            std::to_string(std::size_t{get_index_dimension_of<seq>(arg)}-1) + "]."))} :
          false) , ...);
      }
    }
  }


  /// Get element of matrix arg using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<std::size_t>...I> requires
    element_gettable<Arg, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
  constexpr decltype(auto) get_element(Arg&& arg, const I...i)
  {
    if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>;
    }
    else
    {
      detail::check_index_bounds<false>(arg, std::index_sequence_for<I...> {}, i...);
      return interface::GetElement<std::decay_t<Arg>, I...>::get(std::forward<Arg>(arg), i...);
    }
  }
#else
  template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
    element_gettable<Arg, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...>, int> = 0>
  constexpr decltype(auto) get_element(Arg&& arg, const I...i)
  {
    detail::check_index_bounds<false>(arg, std::index_sequence_for<I...> {}, i...);
    return interface::GetElement<std::decay_t<Arg>, void, I...>::get(std::forward<Arg>(arg), i...);
  }
#endif


  /// Set element to s using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>&> Scalar, std::convertible_to<std::size_t>...I>
    requires element_settable<Arg&, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
  inline Arg& set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::index_sequence_for<I...> {}, i...);
    return interface::SetElement<std::decay_t<Arg>, I...>::set(arg, s, i...);
  }
#else
  template<typename Arg, typename Scalar, typename...I, std::enable_if_t<
    (std::is_convertible_v<I, std::size_t> and ...) and
    std::is_convertible_v<Scalar, const scalar_type_of_t<Arg>&> and
    element_settable<Arg&, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...>, int> = 0>
  inline Arg& set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::index_sequence_for<I...> {}, i...);
    return interface::SetElement<std::decay_t<Arg>, void, I...>::set(arg, s, i...);
  }
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
