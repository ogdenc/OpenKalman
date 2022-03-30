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

  // --------------- //
  //  typed_adapter  //
  // --------------- //

  /**
   * \brief Specifies that T is a typed adapter expression.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_adapter =
#else
  constexpr bool typed_adapter =
#endif
    typed_matrix<T> or covariance<T> or Eigen3::euclidean_expr<T>;


  // ----------------- //
  //  untyped_adapter  //
  // ----------------- //

  /**
   * \brief Specifies that T is an untyped adapter expression.
   * \details Untyped adapter expressions are generally used whenever the native matrix library does not have an
   * important built-in matrix type, such as a single-scalar constant matrix, a diagonal matrix, a triangular matrix,
   * or a hermitian matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_adapter =
#else
  constexpr bool untyped_adapter =
#endif
    Eigen3::eigen_constant_expr<T> or Eigen3::eigen_zero_expr<T> or Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>;


  // ========= //
  //  Aliases  //
  // ========= //

  // ------------------------- //
  //  dense_writable_matrix_t  //
  // ------------------------- //

  namespace detail
  {
    template<typename T, typename...D>
    struct dense_writable_matrix_impl
    {
      using type = std::decay_t<decltype(EquivalentDenseWritableMatrix<T>::make_default(std::declval<const D&>()...))>;
    };


    template<typename T>
    struct dense_writable_matrix_impl<T>
    {
      using type = std::decay_t<decltype(make_dense_writable_matrix_from(std::declval<const T&>()))>;
    };
  }


  /**
    * \brief An alias for a dense, writable matrix, patterned on parameter T.
    * \tparam T A matrix or array from the relevant matrix library.
    * \tparam D Index descriptors defining the dimensions of the new matrix.
    * \todo Create typed Matrix if Ds are typed.
    */
#ifdef __cpp_concepts
  template<indexible T, index_descriptor...D>
#else
  template<typename T, typename...D, std::enable_if_t<indexible<T> and (index_descriptor<D> and ...), int> = 0>
#endif
  using dense_writable_matrix_t = typename detail::dense_writable_matrix_impl<std::decay_t<T>, D...>::type;


  // --------------------------------- //
  //  untyped_dense_writable_matrix_t  //
  // --------------------------------- //

  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam D Index descriptors defining the dimensions of the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, auto...D> requires ((std::is_integral_v<decltype(D)> and D >= 0) and ...)
#else
  template<typename T, auto...D,
    std::enable_if_t<indexible<T> and ((std::is_integral_v<decltype(D)> and D >= 0) and ...), int> = 0>
#endif
  using untyped_dense_writable_matrix_t = dense_writable_matrix_t<T, Dimensions<static_cast<std::size_t>(D)>...>;


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
