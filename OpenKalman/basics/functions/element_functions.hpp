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
    if constexpr (constant_matrix<Arg>)
    {
      return get_scalar_constant_value(constant_coefficient {arg});
    }
    else
    {
      static_assert(element_gettable<Arg&&, I...>, "error in definition of interface::GetElement::get(...)");
      detail::check_index_bounds<false>(arg, std::index_sequence_for<I...> {}, i...);
      return interface::GetElement<std::decay_t<Arg>>::get(std::forward<Arg>(arg), static_cast<const std::size_t>(i)...);
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
    static_assert(element_settable<Arg&&, I...>, "error in definition of interface::SetElement::set(...)");
    detail::check_index_bounds<true>(arg, std::index_sequence_for<I...> {}, i...);
    return interface::SetElement<std::decay_t<Arg>>::set(std::forward<Arg>(arg), s, static_cast<const std::size_t>(i)...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
