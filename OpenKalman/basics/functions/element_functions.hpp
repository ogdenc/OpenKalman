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
    /*template<bool set, typename Arg, std::size_t...I, typename...Ind>
    constexpr void check_index_bounds(const Arg& arg, std::index_sequence<I...>, Ind...i)
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
        ((static_cast<std::size_t>(i) >= get_index_dimension_of<I>(arg) ?
          throw std::out_of_range {(("At least one " + std::string {set ? "s" : "g"} +
            "et_element index out of range:") + ... + (" Index " + std::to_string(I) + " is " +
            std::to_string(i) + " and should be in range [0..." +
            std::to_string(get_index_dimension_of<I>(arg) - 1) + "]."))} :
          false) , ...);
      }
    }*/


    template<typename Arg, std::size_t...I>
    constexpr decltype(auto) get_diag_element(Arg&& arg, std::index_sequence<I...> seq, std::size_t i)
    {
      //detail::check_index_bounds<false>(arg, seq, (I, i)...);
      return interface::Elements<std::decay_t<Arg>>::get(std::forward<Arg>(arg), (I?i:i)...);
    }


    template<typename Arg, typename Scalar, std::size_t...I>
    constexpr Arg&& set_diag_element(Arg&& arg, const Scalar& s, std::index_sequence<I...> seq, std::size_t i)
    {
      //detail::check_index_bounds<true>(arg, seq, (I, i)...);
      return interface::Elements<std::decay_t<Arg>>::set(std::forward<Arg>(arg), s, (I?i:i)...);
    }
  } // namespace detail


/**
 * \brief Get element of matrix arg using I... indices.
 * \tparam Arg
 * \tparam I
 * \param arg
 * \param i
 * \sa element_gettable
 * \return
 */
#ifdef __cpp_lib_concepts
  template<indexible Arg, std::convertible_to<const std::size_t>...I>
  constexpr std::convertible_to<const scalar_type_of_t<Arg>> decltype(auto)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and
    (std::is_convertible_v<I, const std::size_t> and ...), int> = 0>
  constexpr decltype(auto)
#endif
  get_element(Arg&& arg, I...i)
  {
    static_assert(element_gettable<Arg&&, sizeof...(I)>, "Wrong number of indices for get_element");

    if constexpr (constant_matrix<Arg>)
    {
      return get_scalar_constant_value(constant_coefficient {arg});
    }
#ifdef __cpp_lib_concepts
    else if constexpr (requires { interface::Elements<std::decay_t<Arg>>::get(arg, static_cast<const std::size_t>(i)...); })
#else
    else if constexpr (internal::is_element_gettable<Arg, void, I...>::value)
#endif
    {
      //detail::check_index_bounds<false>(arg, std::index_sequence_for<I...> {}, i...);
      return interface::Elements<std::decay_t<Arg>>::get(std::forward<Arg>(arg), static_cast<const std::size_t>(i)...);
    }
    else
    {
      static_assert(sizeof...(I) == 1 and diagonal_matrix<Arg> and max_indices_of_v<Arg> > 1, "Must use correct number of indices");
      std::make_index_sequence<max_indices_of_v<Arg>> seq;
      return detail::get_diag_element(std::forward<Arg>(arg), seq, static_cast<const std::size_t>(i)...);
    }
  }


  /**
   * \brief Set element to s using I... indices.
   * \tparam Arg
   * \tparam I
   * \param arg
   * \param s
   * \param i
   * \sa element_settable
   * \return
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, std::convertible_to<const std::size_t>...I>
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and
    (std::is_convertible_v<I, const std::size_t> and ...), int> = 0>
#endif
  inline Arg&&
  set_element(Arg&& arg, const scalar_type_of_t<Arg>& s, I...i)
  {
    static_assert(element_settable<Arg&&, sizeof...(I)>, "Wrong number of indices for set_element");

#ifdef __cpp_lib_concepts
    if constexpr (requires { interface::Elements<std::decay_t<Arg>>::set(arg, s, static_cast<const std::size_t>(i)...); })
#else
    if constexpr (internal::is_element_settable<Arg, void, I...>::value)
#endif
    {
      //detail::check_index_bounds<true>(arg, std::index_sequence_for<I...> {}, i...);
      return interface::Elements<std::decay_t<Arg>>::set(std::forward<Arg>(arg), s, static_cast<const std::size_t>(i)...);
    }
    else
    {
      static_assert(sizeof...(I) == 1 and diagonal_matrix<Arg> and max_indices_of_v<Arg> > 1, "Must use correct number of indices");
      std::make_index_sequence<max_indices_of_v<Arg>> seq;
      return detail::set_diag_element(std::forward<Arg>(arg), s, seq, static_cast<const std::size_t>(i)...);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
