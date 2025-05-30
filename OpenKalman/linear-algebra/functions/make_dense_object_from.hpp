/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref make_dense_object_from.
 */

#ifndef OPENKALMAN_MAKE_DENSE_OBJECT_FROM_HPP
#define OPENKALMAN_MAKE_DENSE_OBJECT_FROM_HPP

namespace OpenKalman
{
  /**
   * \brief Create a dense, writable matrix from the library of which dummy type T is a member, filled with a set of scalar components.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref Layout::left: column-major;
   * - \ref Layout::right: row-major;
   * - \ref Layout::none (the default): although the elements are listed in row-major order, the layout of the resulting object is unspecified.
   * \tparam T Any dummy type from the relevant library. Its characteristics are ignored.
   * \tparam layout The \ref Layout of Args and the resulting object (\ref Layout::none if unspecified).
   * \tparam Scalar An scalar type for the new matrix. By default, it is the same as T.
   * \tparam Ds \ref coordinates::pattern objects describing the size of the resulting object.
   * \param d_tup A tuple of \ref coordinates::pattern Ds
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, values::number Scalar = scalar_type_of_t<T>, coordinates::pattern...Ds, std::convertible_to<const Scalar> ... Args>
    requires (layout != Layout::stride) and
    (((coordinates::dimension_of_v<Ds> == 0) or ...) ? sizeof...(Args) == 0 :
      (sizeof...(Args) % ((dynamic_pattern<Ds> ? 1 : coordinates::dimension_of_v<Ds>) * ... * 1) == 0))
  inline writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...Ds, typename...Args, std::enable_if_t<
    indexible<T> and values::number<Scalar> and (coordinates::pattern<Ds> and ...) and
    (std::is_convertible_v<Args, const Scalar> and ...) and (layout != Layout::stride) and
    (((coordinates::dimension_of<Ds>::value == 0) or ...) ? sizeof...(Args) == 0 :
      (sizeof...(Args) % ((dynamic_pattern<Ds> ? 1 : coordinates::dimension_of<Ds>::value) * ... * 1) == 0)), int> = 0>
  inline auto
#endif
  make_dense_object_from(const std::tuple<Ds...>& d_tup, Args...args)
  {
    auto m = make_dense_object<T, layout, Scalar>(d_tup);
    if constexpr (sizeof...(Args) > 0)
    {
      constexpr Layout l = layout == Layout::none ? Layout::right : layout;
      return fill_components<l>(m, static_cast<const Scalar>(args)...);
    }
    else return m;
  }


  namespace detail
  {
    template<typename T, std::size_t...Is>
    constexpr bool zero_dimension_count_impl(std::index_sequence<Is...>)
    {
      return ((dimension_size_of_index_is<T, Is, 0> ? 1 : 0) + ... + 0);
    }


    template<typename T>
    struct zero_dimension_count : std::integral_constant<std::size_t,
      zero_dimension_count_impl<T>(std::make_index_sequence<index_count_v<T>>{})> {};


    template<typename T, Layout layout, typename Scalar, std::size_t...I, typename...Args>
    inline auto make_dense_object_from_impl(std::index_sequence<I...>, Args...args)
    {
      std::tuple d_tup {[]{
          if constexpr (dynamic_dimension<T, I>) // There will be only one dynamic dimension, at most.
          {
            constexpr auto dims = ((dynamic_dimension<T, I> ? 1 : index_dimension_of_v<T, I>) * ... * 1);
            if constexpr (dims == 0) return coordinates::Dimensions<0>{};
            else return coordinates::Dimensions<sizeof...(Args) / dims>{};
          }
          else return vector_space_descriptor_of_t<T, I> {};
        }()...};
      return make_dense_object_from<T, layout, Scalar>(d_tup, args...);
    }

  } // namespace detail


  /**
   * \overload
   * \brief Create a dense, writable matrix from a set of components, with size and shape inferred from dummy type T.
   * \details The \ref coordinates::pattern of the result must be unambiguously inferrable from T and the number of indices.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam layout The \ref Layout of Args and the resulting object
   * (\ref Layout::none if unspecified, which means that the values are in \ref Layout::right order but
   * layout of the resulting object is unspecified).
   * \tparam Scalar An scalar type for the new matrix. By default, it is the same as T.
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, values::number Scalar = scalar_type_of_t<T>, std::convertible_to<const Scalar> ... Args>
    requires (layout != Layout::stride) and internal::may_hold_components<T, Args...> and
    (dynamic_index_count_v<T> + detail::zero_dimension_count<T>::value <= 1)
  inline writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename ... Args, std::enable_if_t<
    indexible<T> and values::number<Scalar> and (std::is_convertible_v<Args, const Scalar> and ...) and
    (layout != Layout::stride) and internal::may_hold_components<T, Args...> and
    (dynamic_index_count_v<T> + detail::zero_dimension_count<T>::value <= 1), int> = 0>
  inline auto
#endif
  make_dense_object_from(Args...args)
  {
    constexpr std::make_index_sequence<index_count_v<T>> seq;
    return detail::make_dense_object_from_impl<T, layout, Scalar>(seq, args...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_OBJECT_FROM_HPP
