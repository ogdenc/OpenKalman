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
    template<typename T, std::size_t r, std::size_t c, typename S>
    struct dense_writable_impl_default
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, r, c, S>::make_default());
    };

    template<typename T, std::size_t r, typename S>
    struct dense_writable_impl_default<T, r, dynamic_extent, S>
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, r, dynamic_extent, S>::make_default(2));
    };

    template<typename T, std::size_t c, typename S>
    struct dense_writable_impl_default<T, dynamic_extent, c, S>
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, dynamic_extent, c, S>::make_default(2));
    };

    template<typename T, typename S>
    struct dense_writable_impl_default<T, dynamic_extent, dynamic_extent, S>
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, dynamic_extent, dynamic_extent, S>::make_default(2, 2));
    };

#ifdef __cpp_concepts
    template<typename T, std::size_t r, std::size_t c, typename S>
#else
    template<typename T, std::size_t r, std::size_t c, typename S, typename = void>
#endif
    struct dense_writable_impl
    {
      using type = typename dense_writable_impl_default<T, r, c, S>::type;
    };

#ifdef __cpp_concepts
    template<typename T, std::size_t r, std::size_t c, typename S> requires
      (r == row_extent_of_v<T>) and (c == column_extent_of_v<T>) and std::same_as<S, scalar_type_of_t<T>>
    struct dense_writable_impl<T, r, c, S>
#else
    template<typename T, std::size_t r, std::size_t c, typename S>
    struct dense_writable_impl<T, r, c, S, std::enable_if_t<
      (r == row_extent_of_v<T>) and (c == column_extent_of_v<T>) and std::same_as<S, scalar_type_of_t<T>>>>
#endif
    {
      using type = decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, r, c, S>::convert(std::declval<T>()));
    };
  }

  /**
   * \brief An alias for a self-contained native matrix, based on and equivalent to parameter T.
   * \tparam T The type from which the native matrix is derived.
   * \tparam row_extent Number of rows in the native matrix (defaults to the number of rows in T).
   * \tparam column_extent Number of columns in the native matrix (defaults to the number of columns in T).
   * \tparam scalar_type Scalar type of the matrix (defaults to the Scalar type of T).
   */
  template<typename T,
    std::size_t row_extent =
      dynamic_rows<T> and not dynamic_columns<T> and (self_adjoint_matrix<T> or triangular_matrix<T>) ?
      column_extent_of_v<T> : row_extent_of_v<T>,
    std::size_t column_extent =
      dynamic_columns<T> and not dynamic_rows<T> and (self_adjoint_matrix<T> or triangular_matrix<T>) ?
      row_extent_of_v<T> : column_extent_of_v<T>,
    typename scalar_type = scalar_type_of_t<T>>
  using equivalent_dense_writable_matrix_t =
    std::decay_t<typename detail::dense_writable_impl<std::decay_t<T>, row_extent, column_extent, scalar_type>::type>;


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

  // ---------------------- //
  //  constant_coefficient  //
  // ---------------------- //

  /**
   * \brief A typed matrix or covariance is a constant matrix if its nested matrix is a constant matrix.
   * \details In the case of a triangular_covariance, the nested matrix must also be a zero_matrix.
   */
#ifdef __cpp_concepts
  template<typename T> requires typed_matrix<T> or self_adjoint_covariance<T> or
    (triangular_covariance<T> and zero_matrix<T>)
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<typed_matrix<T> or self_adjoint_covariance<T> or
      (triangular_covariance<T> and zero_matrix<nested_matrix_of<T>>)>>
#endif
    : internal::constant_coefficient_type<constant_coefficient_v<nested_matrix_of<T>>> {};


  // ------------------------------- //
  //  constant_diagonal_coefficient  //
  // ------------------------------- //

  /**
   * \brief A typed_matrix or covariance is a constant_diagonal_matrix if its nested matrix is a constant_diagonal_matrix.
   */
#ifdef __cpp_concepts
  template<typename T> requires typed_matrix<T> or covariance<T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
#endif
    : internal::constant_coefficient_type<constant_diagonal_coefficient_v<nested_matrix_of<T>>> {};


  namespace internal
  {
    // -------------------- //
    //  is_diagonal_matrix  //
    // -------------------- //

    /**
     * A covariance is a diagonal matrix if its nested matrix is diagonal
     */
#ifdef __cpp_concepts
    template<covariance T>
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<covariance<T>>>
#endif
      : std::bool_constant<diagonal_matrix<nested_matrix_of<T>>> {};


    /**
     * A typed matrix is diagonal if its nested matrix is diagonal and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T>
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : std::bool_constant<diagonal_matrix<nested_matrix_of<T>> and
        equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>> {};


    // ------------------------------ //
    //  is_lower_self_adjoint_matrix  //
    // ------------------------------ //

    /**
     * A self_adjoint_covariance is lower-self-adjoint if it its covariance is lower-triangular or lower-self-adjoint.
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance T>
    struct is_lower_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_lower_self_adjoint_matrix<T, std::enable_if_t<self_adjoint_covariance<T>>>
#endif
      : std::bool_constant<lower_self_adjoint_matrix<nested_matrix_of<T>> or
        lower_triangular_matrix<nested_matrix_of<T>>> {};


    /**
     * A typed matrix is lower-self-adjoint if its nested matrix is lower-self-adjoint and
     * its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T>
    struct is_lower_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_lower_self_adjoint_matrix<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : std::bool_constant<lower_self_adjoint_matrix<nested_matrix_of<T>> and
        equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>> {};


  // ------------------------------ //
  //  is_upper_self_adjoint_matrix  //
  // ------------------------------ //

  /**
   * A self_adjoint_covariance is upper-self-adjoint if it its covariance is upper-triangular or upper-self-adjoint.
   */
#ifdef __cpp_concepts
  template<self_adjoint_covariance T>
  struct is_upper_self_adjoint_matrix<T>
#else
  template<typename T>
  struct is_upper_self_adjoint_matrix<T, std::enable_if_t<self_adjoint_covariance<T>>>
#endif
    : std::bool_constant<upper_self_adjoint_matrix<std::decay_t<nested_matrix_of<T>>> or
      upper_triangular_matrix<std::decay_t<nested_matrix_of<T>>>> {};


  /**
   * A typed matrix is upper-self-adjoint if its nested matrix is upper-self-adjoint and
   * its row and column coefficients are equivalent.
   */
#ifdef __cpp_concepts
  template<typed_matrix T>
  struct is_upper_self_adjoint_matrix<T>
#else
  template<typename T>
  struct is_upper_self_adjoint_matrix<T, std::enable_if_t<typed_matrix<T>>>
#endif
    : std::bool_constant<upper_self_adjoint_matrix<std::decay_t<nested_matrix_of<T>>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>> {};


    // ---------------------------- //
    //  is_lower_triangular_matrix  //
    // ---------------------------- //

    /**
     * A triangular_covariance is lower-triangular if its nested matrix is lower_triangular or lower_self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<triangular_covariance T>
    struct is_lower_triangular_matrix<T>
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<triangular_covariance<T>>>
#endif
      : std::bool_constant<lower_triangular_matrix<nested_matrix_of<T>> or
        lower_self_adjoint_matrix<nested_matrix_of<T>>> {};


    /**
     * A typed matrix is lower_triangular if its nested matrix is lower_triangular.
     */
#ifdef __cpp_concepts
    template<typed_matrix T>
    struct is_lower_triangular_matrix<T>
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : std::bool_constant<lower_triangular_matrix<nested_matrix_of<T>> and
        equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>> {};


    // ---------------------------- //
    //  is_upper_triangular_matrix  //
    // ---------------------------- //

    /**
     * A triangular_covariance is upper-triangular based on its triangle type.
     */
#ifdef __cpp_concepts
    template<triangular_covariance T>
    struct is_upper_triangular_matrix<T>
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<triangular_covariance<T>>>
#endif
      : std::bool_constant<upper_triangular_matrix<nested_matrix_of<T>> or
        upper_self_adjoint_matrix<nested_matrix_of<T>>> {};


    /**
     * A typed matrix is upper_triangular if its nested matrix is upper_triangular.
     */
#ifdef __cpp_concepts
    template<typed_matrix T>
    struct is_upper_triangular_matrix<T>
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : std::bool_constant<upper_triangular_matrix<nested_matrix_of<T>> and
        equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>> {};


  } // namespace internal


  // ------------- //
  //  is_writable  //
  // ------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typed_matrix T> requires writable<nested_matrix_of<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<typed_matrix<T> and writable<nested_matrix_of<T>>>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<covariance T> requires writable<nested_matrix_of<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<covariance<T> and writable<nested_matrix_of<T>>>> : std::true_type {};
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
      (requires { typename nested_matrix_of<T>; } and has_const<nested_matrix_of<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_of<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct has_same_matrix_shape : std::false_type {};


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      (dynamic_rows<T> or dynamic_rows<U> or row_extent_of_v<T> == row_extent_of_v<U>) and
      (dynamic_columns<T> or dynamic_columns<U> or column_extent_of_v<T> == column_extent_of_v<U>) and
      (std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>)
    struct has_same_matrix_shape<T, U> : std::true_type {};
#else
    template<typename T, typename U>
    struct has_same_matrix_shape<T, U, std::enable_if_t<
      (dynamic_rows<T> or dynamic_rows<U> or row_extent_of<T>::value == row_extent_of<U>::value) and
      (dynamic_columns<T> or dynamic_columns<U> or column_extent_of<T>::value == column_extent_of<U>::value) and
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
