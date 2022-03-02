/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  using namespace interface;

  // ========= //
  //  Aliases  //
  // ========= //

  // --------------------------- //
  //  equivalent_self_contained  //
  // --------------------------- //

  namespace detail
  {

    template<typename Trait, std::size_t r, std::size_t c>
    struct dense_writable_impl_default
    {
      using type = decltype(Trait::make_default());
    };

    template<typename Trait, std::size_t r>
    struct dense_writable_impl_default<Trait, r, dynamic_size>
    {
      using type = decltype(Trait::make_default(2));
    };

    template<typename Trait, std::size_t c>
    struct dense_writable_impl_default<Trait, dynamic_size, c>
    {
      using type = decltype(Trait::make_default(2));
    };

    template<typename Trait>
    struct dense_writable_impl_default<Trait, dynamic_size, dynamic_size>
    {
      using type = decltype(Trait::make_default(2, 2));
    };


#ifdef __cpp_concepts
    template<typename T, std::size_t r, std::size_t c, typename S>
#else
    template<typename T, std::size_t r, std::size_t c, typename S, typename = void>
#endif
    struct dense_writable_impl
    {
      using Trait = EquivalentDenseWritableMatrix<std::decay_t<T>, r, c, S>;
      using type = typename dense_writable_impl_default<Trait, r, c>::type;
    };


#ifdef __cpp_concepts
    template<typename T, std::size_t r, std::size_t c, typename S> requires
      (r == row_dimension_of_v<T>) and (c == column_dimension_of_v<T>) and std::same_as<S, scalar_type_of_t<T>>
    struct dense_writable_impl<T, r, c, S>
#else
    template<typename T, std::size_t r, std::size_t c, typename S>
    struct dense_writable_impl<T, r, c, S, std::enable_if_t<(r == row_dimension_of<T>::value) and
      (c == column_dimension_of<T>::value) and std::is_same_v<S, scalar_type_of<T>::type>>>
#endif
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, r, c, S>::convert(std::declval<T>()));
    };

  } // namespace detail

  /**
   * \brief An alias for a self-contained native matrix, based on and equivalent to parameter T.
   * \tparam T The type from which the native matrix is derived.
   * \tparam rows Number of rows in the native matrix (defaults to the number of rows in T).
   * \tparam columns Number of columns in the native matrix (defaults to the number of columns in T).
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   */
  template<typename T,
    std::size_t rows = dynamic_rows<T> and not dynamic_columns<T> and (self_adjoint_matrix<T> or triangular_matrix<T>) ?
      column_dimension_of_v<T> : row_dimension_of_v<T>,
    std::size_t columns = dynamic_columns<T> and not dynamic_rows<T> and (self_adjoint_matrix<T> or triangular_matrix<T>) ?
      row_dimension_of_v<T> : column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>>
  using equivalent_dense_writable_matrix_t =
    std::decay_t<typename detail::dense_writable_impl<std::decay_t<T>, rows, columns, Scalar>::type>;


  // --------------------------- //
  //  equivalent_self_contained  //
  // --------------------------- //

  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   * \sa self_contained, make_self_contained
   * \internal \sa interface::Dependencies
   */
  template<typename T>
  using equivalent_self_contained_t = std::remove_reference_t<decltype(make_self_contained(std::declval<T>()))>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \tparam T The type in question.
   * \details A passable type T is either an lvalue reference or is \ref equivalent_self_contained_t.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, equivalent_self_contained_t<T>>;


  // ========================================================================== //
  //  Traits for which specializations must be defined in the matrix interface  //
  // ========================================================================== //

  // ------------- //
  //  is_writable  //
  // ------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typed_matrix T> requires writable<nested_matrix_of_t<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<typed_matrix<T> and writable<nested_matrix_of_t<T>>>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<covariance T> requires writable<nested_matrix_of_t<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<covariance<T> and writable<nested_matrix_of_t<T>>>> : std::true_type {};
#endif

  }  // namespace internal


  // --------------- //
  //  is_modifiable  //
  // --------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct has_const : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires std::is_const_v<std::remove_reference_t<T>> or
      (requires { typename nested_matrix_of_t<T>; } and has_const<nested_matrix_of_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_of_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct has_same_matrix_shape : std::false_type {};


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      (dynamic_rows<T> or dynamic_rows<U> or row_dimension_of_v<T> == row_dimension_of_v<U>) and
      (dynamic_columns<T> or dynamic_columns<U> or column_dimension_of_v<T> == column_dimension_of_v<U>) and
      (std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>)
    struct has_same_matrix_shape<T, U> : std::true_type {};
#else
    template<typename T, typename U>
    struct has_same_matrix_shape<T, U, std::enable_if_t<
      (dynamic_rows<T> or dynamic_rows<U> or row_dimension_of<T>::value == row_dimension_of<U>::value) and
      (dynamic_columns<T> or dynamic_columns<U> or column_dimension_of<T>::value == column_dimension_of<U>::value) and
      (std::is_same_v<typename scalar_type_of<T>::type, typename scalar_type_of<U>::type>)>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not has_same_matrix_shape<T, U>::value) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (upper_triangular_matrix<T> and not upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and not lower_triangular_matrix<U>) or
      (self_adjoint_matrix<T> and not self_adjoint_matrix<U>)
    struct is_modifiable<T, U> : std::false_type {};
#else
    template<typename T, typename U>
    struct is_modifiable<T, U, std::enable_if_t<
      has_const<T>::value or
      (not has_same_matrix_shape<T, U>::value) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (upper_triangular_matrix<T> and not upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and not lower_triangular_matrix<U>) or
      (self_adjoint_matrix<T> and not self_adjoint_matrix<U>)>> : std::false_type {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_TRAITS_HPP
