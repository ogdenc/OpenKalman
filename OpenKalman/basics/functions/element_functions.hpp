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
      return interface::indexible_object_traits<std::decay_t<Arg>>::get(std::forward<Arg>(arg), (I ? i : i)...);
    }
  } // namespace detail


/**
 * \brief Get element of matrix arg using I... indices.
 * \details The number of indices I... may be 0 if there is only one element, or 1 if Arg is diagonal.
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
    constexpr auto N = sizeof...(I);

    if constexpr (constant_matrix<Arg> and (not one_by_one_matrix<Arg>) and (N == index_count_v<Arg> or N == 0))
    {
      return get_scalar_constant_value(constant_coefficient {arg});
    }
    else if constexpr (internal::is_element_gettable<Arg, N>::value)
    {
      //detail::check_index_bounds<false>(arg, std::index_sequence_for<I...> {}, i...);
      return interface::indexible_object_traits<std::decay_t<Arg>>::get(std::forward<Arg>(arg), static_cast<const std::size_t>(i)...);
    }
    else if constexpr (N == 1 and diagonal_matrix<Arg, Likelihood::maybe> and index_count_v<Arg> >= 1)
    {
      if constexpr (not diagonal_matrix<Arg>) if (not get_is_square(arg))
        throw std::invalid_argument {"Wrong number of indices in arguments to get_element."};
      std::make_index_sequence<index_count_v<Arg>> seq;
      return detail::get_diag_element(std::forward<Arg>(arg), seq, static_cast<const std::size_t>(i)...);
    }
    else
    {
      static_assert(N == 0, "Must use correct number of indices");
      static_assert(one_by_one_matrix<Arg, Likelihood::maybe>, "Calling get_element without indices only allowed for one-by-one matrices.");
      if constexpr (not one_by_one_matrix<Arg>) if (get_index_dimension_of<0>(arg) != 1 or get_index_dimension_of<1>(arg) != 1)
        throw std::invalid_argument {"Wrong number of indices in arguments to get_element."};
      std::make_index_sequence<index_count_v<Arg>> seq;
      return detail::get_diag_element(std::forward<Arg>(arg), seq, static_cast<const std::size_t>(0));
    }
  }


  namespace detail
  {
    template<typename Arg, typename Scalar, std::size_t...I>
    constexpr void set_diag_element(Arg& arg, const Scalar& s, std::index_sequence<I...> seq, std::size_t i)
    {
      //detail::check_index_bounds<true>(arg, seq, (I, i)...);
      interface::indexible_object_traits<std::decay_t<Arg>>::set(arg, s, (I ? i : i)...);
    }
  } // namespace detail


  /**
   * \brief Set element to s using I... indices.
   * \details The number of indices I... may be 0 if there is only one element, or 1 if Arg is diagonal.
   * \tparam Arg
   * \tparam I
   * \param arg
   * \param s
   * \param i
   * \sa element_settable
   * \return
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, std::convertible_to<const std::size_t>...I> requires (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and
    (std::is_convertible_v<I, const std::size_t> and ...) and not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&&
  set_element(Arg&& arg, const scalar_type_of_t<Arg>& s, I...i)
  {
    constexpr auto N = sizeof...(I);

    if constexpr (internal::is_element_settable<Arg, N>::value)
    {
      //detail::check_index_bounds<true>(arg, std::index_sequence_for<I...> {}, i...);
      interface::indexible_object_traits<std::decay_t<Arg>>::set(arg, s, static_cast<const std::size_t>(i)...);
    }
    else if constexpr (N == 0 and one_by_one_matrix<Arg, Likelihood::maybe>)
    {
      if constexpr (not one_by_one_matrix<Arg>) if (get_index_dimension_of<0>(arg) != 1 or get_index_dimension_of<1>(arg) != 1)
        throw std::invalid_argument {"Wrong number of indices in arguments to set_element."};
      std::make_index_sequence<index_count_v<Arg>> seq;
      detail::set_diag_element(arg, s, seq, static_cast<const std::size_t>(0));
    }
    else
    {
      static_assert(N == 1 and diagonal_matrix<Arg, Likelihood::maybe> and index_count_v<Arg> > 1,
        "Must use correct number of indices");
      if constexpr (not diagonal_matrix<Arg>)
        if (not get_is_square(arg))
          throw std::invalid_argument{"Wrong number of indices in arguments to set_element."};
      std::make_index_sequence<index_count_v<Arg>> seq;
      detail::set_diag_element(arg, s, seq, static_cast<const std::size_t>(i)...);
    }
    return std::forward<Arg>(arg);
  }


  namespace internal
  {
    namespace detail
    {
      template<typename T, std::size_t N, std::size_t...I>
      constexpr bool may_hold_components_impl(std::index_sequence<I...>)
      {
        constexpr auto dims = ((dynamic_dimension<T, I> ? 1 : index_dimension_of_v<T, I>) * ... * 1);
        if constexpr (N == 0) return dims == 0;
        else if constexpr (dims == 0) return false;
        else return N % dims == 0;
      }
    } // namespace detail


    template<typename T, typename...Components>
#ifdef __cpp_concepts
    concept may_hold_components = indexible<T> and (std::convertible_to<Components, const scalar_type_of_t<T>> and ...) and
#else
    constexpr bool may_hold_components = indexible<T> and (std::is_convertible_v<Components, const scalar_type_of_t<T>> and ...) and
#endif
      detail::may_hold_components_impl<T, sizeof...(Components)>(std::make_index_sequence<index_count_v<T>> {});

  } // namespace internal


  /**
   * \overload
   * \brief Set all the components of an object from a list of scalar values.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref Layout::left: column-major;
   * - \ref Layout::right: row-major (the default).
   * \tparam layout The \ref Layout of Args and the resulting object (\ref Layout::right if unspecified).
   * \param arg The object to be modified.
   * \param s Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::right, writable Arg, scalar_type ... S>
    requires (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>
  inline writable auto
#else
  template<Layout layout = Layout::right, typename Arg, typename...S, std::enable_if_t<
    writable<Arg> and (scalar_type<S> and ...) and
    (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>, int> = 0>
  inline Arg&&
#endif
  set_elements(Arg&& arg, S...s)
  {
    if constexpr (sizeof...(S) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = scalar_type_of_t<Arg>;
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return Trait::template fill_with_elements<layout>(std::forward<Arg>(arg), static_cast<const Scalar>(s)...);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_FUNCTIONS_HPP
