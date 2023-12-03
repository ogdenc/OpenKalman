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
 * \brief Overloaded general functions for making dense writable objects.
 */

#ifndef OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP
#define OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP

namespace OpenKalman
{

  namespace detail
  {
    template<typename M, typename Arg, typename...J>
    static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<>, J...j)
    {
      set_element(m, get_element(arg, j...), j...);
    }


    template<typename M, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); i++)
        copy_tensor_elements(m, arg, std::index_sequence<Is...> {}, j..., i);
    }
  } // namespace detail


  /**
   * \brief Convert the argument to a dense, writable matrix of a particular scalar type.
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, the interface will decide the layout.
   * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
   * \param arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<Layout layout, scalar_type Scalar, indexible Arg> requires (layout != Layout::stride)
  constexpr writable decltype(auto)
#else
  template<Layout layout, typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg> and
    (layout != Layout::stride), int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    using M = std::decay_t<decltype(make_default_dense_writable_matrix_like<layout, Scalar>(arg))>;

    if constexpr (writable<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (std::is_constructible_v<M, Arg&&>)
    {
      M m {std::forward<Arg>(arg)};
      return m;
    }
    else if constexpr (std::is_constructible_v<M, decltype(to_native_matrix<M>(std::declval<Arg&&>()))>)
    {
      M m {to_native_matrix<M>(std::forward<Arg>(arg))};
      return m;
    }
    else
    {
      auto m {make_default_dense_writable_matrix_like<layout, Scalar>(arg)};
      if constexpr (std::is_assignable_v<M&, Arg&&>)
      {
        m = std::forward<Arg>;
      }
      else if constexpr (std::is_assignable_v<M&, decltype(to_native_matrix<M>(std::declval<Arg&&>()))>)
      {
        m = to_native_matrix<M>(std::forward<Arg>);
      }
      else
      {
        detail::copy_tensor_elements(m, arg, std::make_index_sequence<index_count_v<Arg>>{});
      }
      return m;
    }
  }


  /**
   * \overload
   * \brief Convert the argument to a dense, writable matrix with the same scalar type as the argument.
   * \tparam layout The \ref Layout of the resulting object (optional). If this is omitted or Layout::none,
   * the interface will decide the layout.
   * \param arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::none, indexible Arg> requires (layout != Layout::stride)
  constexpr writable decltype(auto)
#else
  template<Layout layout = Layout::none, typename Arg, std::enable_if_t<indexible<Arg> and (layout != Layout::stride), int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    return make_dense_writable_matrix_from<layout, scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Create a dense, writable matrix from the library of which dummy type T is a member, filled with a set of scalar components.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref Layout::left: column-major;
   * - \ref Layout::right: row-major;
   * - \ref Layout::none (the default): although the elements are listed in row-major order, the layout of the resulting object is unspecified.
   * \tparam T Any dummy type from the relevant library. Its characteristics are ignored.
   * \tparam layout The \ref Layout of Args and the resulting object (\ref Layout::none if unspecified).
   * \tparam Scalar An scalar type for the new matrix. By default, it is the same as T.
   * \tparam Ds \ref vector_space_descriptor objects describing the size of the resulting object.
   * \param d_tup A tuple of \ref vector_space_descriptor Ds
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor...Ds, std::convertible_to<const Scalar> ... Args>
    requires (layout != Layout::stride) and
    (((dimension_size_of_v<Ds> == 0) or ...) ? sizeof...(Args) == 0 :
      (sizeof...(Args) % ((dynamic_vector_space_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0))
  inline writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...Ds, typename...Args, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (vector_space_descriptor<Ds> and ...) and
    (std::is_convertible_v<Args, const Scalar> and ...) and (layout != Layout::stride) and
    (((dimension_size_of<Ds>::value == 0) or ...) ? sizeof...(Args) == 0 :
      (sizeof...(Args) % ((dynamic_vector_space_descriptor<Ds> ? 1 : dimension_size_of<Ds>::value) * ... * 1) == 0)), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(const std::tuple<Ds...>& d_tup, Args...args)
  {
    auto m = std::apply([](const auto&...d) { return make_default_dense_writable_matrix_like<T, layout, Scalar>(d...); }, d_tup);
    if constexpr (sizeof...(Args) > 0)
    {
      constexpr Layout l = layout == Layout::none ? Layout::right : layout;
      return set_elements<l>(std::move(m), static_cast<const Scalar>(args)...);
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
    inline auto make_dense_writable_matrix_from_impl(std::index_sequence<I...>, Args...args)
    {
      std::tuple d_tup {[]{
          if constexpr (dynamic_dimension<T, I>) // There will be only one dynamic dimension, at most.
          {
            constexpr auto dims = ((dynamic_dimension<T, I> ? 1 : index_dimension_of_v<T, I>) * ... * 1);
            if constexpr (dims == 0) return Dimensions<0>{};
            else return Dimensions<sizeof...(Args) / dims>{};
          }
          else return vector_space_descriptor_of_t<T, I> {};
        }()...};
      return make_dense_writable_matrix_from<T, layout, Scalar>(d_tup, args...);
    }
  } // namespace detail


  /**
   * \overload
   * \brief Create a dense, writable matrix from a set of components, with size and shape inferred from dummy type T.
   * \details The \ref vector_space_descriptor of the result must be unambiguously inferrable from T and the number of indices.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam layout The \ref Layout of Args and the resulting object
   * (\ref Layout::none if unspecified, which means that the values are in \ref Layout::right order but
   * layout of the resulting object is unspecified).
   * \tparam Scalar An scalar type for the new matrix. By default, it is the same as T.
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>, std::convertible_to<const Scalar> ... Args>
    requires (layout != Layout::stride) and internal::may_hold_components<T, Args...> and
    (dynamic_index_count_v<T> + detail::zero_dimension_count<T>::value <= 1)
  inline writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename ... Args, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (std::is_convertible_v<Args, const Scalar> and ...) and
    (layout != Layout::stride) and internal::may_hold_components<T, Args...> and
    (dynamic_index_count_v<T> + detail::zero_dimension_count<T>::value <= 1), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(Args...args)
  {
    constexpr std::make_index_sequence<index_count_v<T>> seq;
    return detail::make_dense_writable_matrix_from_impl<T, layout, Scalar>(seq, args...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP
