/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file Traits.h
 * A header file containing forward declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * Type trait testing whether T is self-contained (i.e., can be the return value of a function)
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_self_contained : std::false_type {};
  }


  /**
   * T is a self-contained matrix or expression (i.e., it can be created in a function and returned as the result).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_contained = internal::is_self_contained<std::decay_t<T>>::value;
#else
  inline constexpr bool self_contained = internal::is_self_contained<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A typed matrix or covariance is self-contained if its base matrix is self-contained and not a reference.
     */
#ifdef __cpp_concepts
    template<typename T> requires (typed_matrix<T> or covariance<T>) and
      self_contained<typename MatrixTraits<T>::BaseMatrix> and
      (not std::is_reference_v<typename MatrixTraits<T>::BaseMatrix>)
    struct is_self_contained<T> : std::true_type {};
#else
      template<typename T>
    struct is_self_contained<T, std::enable_if_t<(typed_matrix<T> or covariance<T>) and
      (not std::is_reference_v<typename MatrixTraits<T>::BaseMatrix>)>>
      : is_self_contained<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A distribution is self-contained if its associated mean and covariance are also self-contained.
     */
#ifdef __cpp_concepts
    template<distribution T> requires self_contained<typename DistributionTraits<T>::Mean> and
      self_contained<typename DistributionTraits<T>::Covariance>
    struct is_self_contained<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<distribution<T>>>
      : std::bool_constant<self_contained<typename DistributionTraits<T>::Mean> and
        self_contained<typename DistributionTraits<T>::Covariance>> {};
#endif

  } // namespace internal


  // ------------------ //
  //  self_contained_t  //
  // ------------------ //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct self_contained_impl { using type = typename MatrixTraits<T>::SelfContained; };


#ifdef __cpp_concepts
    template<distribution T>
    struct self_contained_impl<T>
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<distribution<T>>>
#endif
    {
      using type = typename DistributionTraits<T>::SelfContained;
    };
  }


  /**
   * A type, based on T, that is transformed into a self-contained type if necessary.
   */
  template<typename T>
  using self_contained_t = std::conditional_t<
    self_contained<T>, std::decay_t<T>, typename detail::self_contained_impl<T>::type>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * A type, based on T, that is passable to a function as a parameter.
   * (i.e., it is either an lvalue reference or an rvalue reference to a self-contained type)
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, self_contained_t<T>>;


  // --------------- //
  //  native_matrix  //
  // --------------- //

  /**
   * A type, based on T, that is a self-contained native matrix.
   * \tparam rows Number of rows in the native matrix (defaults to number of rows in T).
   * \tparam cols Number of columns in the native matrix (default to number of columns in T).
   */
  template<typename T, std::size_t rows = MatrixTraits<T>::dimension, std::size_t cols = MatrixTraits<T>::columns>
  using native_matrix_t = typename MatrixTraits<T>::template NativeMatrix<rows, cols>;


  /// Make a self-contained, native matrix based on the shape of M from a list of coefficients in row-major order.
#ifdef __cpp_concepts
  template<typename M, std::convertible_to<typename MatrixTraits<M>::Scalar> ... Args> requires
    (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns)
#else
  template<typename M, typename ... Args, std::enable_if_t<
    (std::is_convertible_v<Args, typename MatrixTraits<M>::Scalar> and ...) and
    (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns), int> = 0>
#endif
  static auto
  make_native_matrix(const Args ... args)
  {
    return MatrixTraits<native_matrix_t<M>>::make(args...);
  }


  // ============================================= //
  //  Traits to be defined in the matrix interface  //
  // ============================================= //

  // ---------------- //
  //  is_zero_matrix  //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is a zero matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_zero_matrix : std::false_type {};
  }


  /**
   * T is a zero matrix.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A typed matrix or covariance is a zero matrix if its base matrix is a zero matrix.
     */
#ifdef __cpp_concepts
    template<typename T> requires (typed_matrix<T> or covariance<T>) and
      zero_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_zero_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<(typed_matrix<T> or covariance<T>)>>
      : is_zero_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A distribution is a zero matrix if its associated mean and covariance are also zero matrices.
     */
#ifdef __cpp_concepts
    template<distribution T> requires zero_matrix<typename DistributionTraits<T>::Mean> and
      zero_matrix<typename DistributionTraits<T>::Covariance>
    struct is_zero_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<distribution<T>>>
      : std::bool_constant<zero_matrix<typename DistributionTraits<T>::Mean> and
        zero_matrix<typename DistributionTraits<T>::Covariance>> {};
#endif
  }


  // -------------------- //
  //  is_identity_matrix  //
  // -------------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is an identity matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_identity_matrix : std::false_type {};
  }

  /**
   * T is an identity matrix.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A covariance is an identity matrix if its base matrix is an identity matrix.
     */
#ifdef __cpp_concepts
    template<covariance T> requires identity_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_identity_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<covariance<T>>>
      : is_identity_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A typed matrix is an identity matrix if its base matrix is an identity matrix
     * and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires identity_matrix<typename MatrixTraits<T>::BaseMatrix> and
    equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_identity_matrix<T> : std::true_type {};
#else
      template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_identity_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif
  }


  // ------------------- //
  //  one_by_one_matrix  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_1by1 : std::false_type {};

    template<typename T>
    struct is_1by1<T, std::enable_if_t<(MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1)>>
      : std::true_type {};
  }
#endif


  /**
   * T is a one-by-one matrix (i.e., one row and one column).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept one_by_one_matrix = (MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1);
#else
  inline constexpr bool one_by_one_matrix = detail::is_1by1<T>::value;
#endif


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is a diagonal matrix
     * (other than zero_matrix, identity_matrix, or one_by_one_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_diagonal_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_diag_matrix_impl : std::false_type {};

    template<typename T>
    struct is_diag_matrix_impl<T, std::enable_if_t<internal::is_diagonal_matrix<std::decay_t<T>>::value or
      zero_matrix<T> or identity_matrix<T> or one_by_one_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * T is a diagonal matrix.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept diagonal_matrix = internal::is_diagonal_matrix<std::decay_t<T>>::value or
    zero_matrix<T> or identity_matrix<T> or one_by_one_matrix<T>;
#else
  template<typename T>
  inline constexpr bool diagonal_matrix = detail::is_diag_matrix_impl<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A covariance is a diagonal matrix if its base matrix is diagonal
     */
#ifdef __cpp_concepts
    template<covariance T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<covariance<T>>>
      : is_diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A typed matrix is diagonal if its base matrix is diagonal and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A distribution is diagonal if its covariance matrix is diagonal.
     */
#ifdef __cpp_concepts
    template<distribution T> requires diagonal_matrix<typename DistributionTraits<T>::Covariance>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<distribution<T>>>
      : std::bool_constant<diagonal_matrix<typename DistributionTraits<T>::Covariance>> {};
#endif
  }


  // ------------------------ //
  //  is_self_adjoint_matrix  //
  // ------------------------ //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is a self-adjoint matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_self_adjoint_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_sa_matrix_impl : std::false_type {};

    template<typename T>
    struct is_sa_matrix_impl<T, std::enable_if_t<
      internal::is_self_adjoint_matrix<std::decay_t<T>>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * T is a self-adjoint matrix.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept self_adjoint_matrix = internal::is_self_adjoint_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool self_adjoint_matrix = detail::is_sa_matrix_impl<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A covariance matrix is self-adjoint if it is not a square root (Cholesky) covariance.
     */
#ifdef __cpp_concepts
    template<covariance T> requires (not square_root_covariance<T>)
    struct is_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<covariance<T> and (not square_root_covariance<T>)>>
#endif
      : std::true_type {};


    /**
     * \internal
     * A typed matrix is self-adjoint if its base matrix is self-adjoint and
     * its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_self_adjoint_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix> {};
#endif


    /**
     * \internal
     * A distribution is self-adjoint if its associated covariance matrix is self-adjoint.
     */
#ifdef __cpp_concepts
    template<distribution T> requires self_adjoint_matrix<typename DistributionTraits<T>::Covariance>
    struct is_self_adjoint_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<distribution<T>>>
      : is_self_adjoint_matrix<typename DistributionTraits<T>::Covariance> {};
#endif
  }


  // ---------------------------- //
  //  is_lower_triangular_matrix  //
  // ---------------------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is a lower-triangular matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_lower_triangular_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_lt_matrix_impl : std::false_type {};

    template<typename T>
    struct is_lt_matrix_impl<T, std::enable_if_t<
      internal::is_lower_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * T is a lower-triangular matrix.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept lower_triangular_matrix = internal::is_lower_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool lower_triangular_matrix = detail::is_lt_matrix_impl<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A square root covariance is lower-triangular if its base matrix is either lower-triangular or self-adjoint.
     * If the base matrix is self-adjoint, the square root covariance can be either lower- or upper-triangular.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires lower_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_lower_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
      : std::bool_constant<lower_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
          self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>> {};
#endif
  }


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether an object is an upper-triangular matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_upper_triangular_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_ut_matrix_impl : std::false_type {};

    template<typename T>
    struct is_ut_matrix_impl<T, std::enable_if_t<
      internal::is_upper_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * T is an upper-triangular matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept upper_triangular_matrix = internal::is_upper_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool upper_triangular_matrix = detail::is_ut_matrix_impl<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A square root covariance is upper-triangular if its base matrix is either upper-triangular or self-adjoint.
     * If the base matrix is self-adjoint, the square root covariance can be either lower- or upper-triangular.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires upper_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
        : std::bool_constant<upper_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
            self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>> {};
#endif
  }


  // ------------------- //
  //  triangular_matrix  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_triangular_matrix_impl : std::false_type {};

    template<typename T>
    struct is_triangular_matrix_impl<T, std::enable_if_t<lower_triangular_matrix<T> or upper_triangular_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * T is a triangular matrix (upper or lower).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept triangular_matrix = lower_triangular_matrix<T> or upper_triangular_matrix<T>;
#else
  template<typename T>
  inline constexpr bool triangular_matrix = detail::is_triangular_matrix_impl<T>::value;
#endif


  // ----------------------- //
  //  same_triangle_type_as  //
  // ----------------------- //

  namespace internal
  {
    /**
     * T and U have the same triangular type (upper or lower).
     * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T, typename U>
#ifdef __cpp_concepts
    concept same_triangle_type_as =
      (upper_triangular_matrix<T> and upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and lower_triangular_matrix<U>);
#else
    inline constexpr bool same_triangle_type_as =
      (upper_triangular_matrix<T> and upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and lower_triangular_matrix<U>);
#endif
  }


  // ------------------ //
  //  is_cholesky_form  //
  // ------------------ //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether the base matrix is a Cholesky square root.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_cholesky_form : std::false_type {};
  }


  /**
   * T has a base matrix that is a Cholesky square root.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#else
  inline constexpr bool cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * A covariance has a Cholesky form if its base matrix is not self-adjoint.
     */
#ifdef __cpp_concepts
    template<covariance T> requires (not self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>)
    struct is_cholesky_form<T> : std::true_type {};
#else
    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<covariance<T>>>
      : std::bool_constant<not self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>> {};
#endif


    /**
     * \internal
     * A distribution has a Cholesky form if its associated Covariance has a Cholesky form.
     */
#ifdef __cpp_concepts
    template<distribution T> requires cholesky_form<typename DistributionTraits<T>::Covariance>
    struct is_cholesky_form<T> : std::true_type {};
#else
    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<distribution<T>>>
      : is_cholesky_form<typename DistributionTraits<T>::Covariance> {};
#endif
  }


  // -------------------------------- //
  //  TriangleType, triangle_type_of  //
  // -------------------------------- //

  /**
   * The type of a triangular matrix, either lower, upper, or diagonal.
   */
  enum struct TriangleType { lower, upper, diagonal };


  namespace detail
  {
#ifdef __cpp_concepts
    template<triangular_matrix T>
#else
    template<typename T, typename Enable = void>
#endif
    struct triangle_type_of_impl : std::integral_constant<TriangleType, TriangleType::lower> {};


#ifdef __cpp_concepts
    template<lower_triangular_matrix T> requires (not diagonal_matrix<T>)
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<lower_triangular_matrix<T> and not diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::lower> {};


#ifdef __cpp_concepts
    template<upper_triangular_matrix T> requires (not diagonal_matrix<T>)
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<upper_triangular_matrix<T> and not diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::upper> {};


#ifdef __cpp_concepts
    template<diagonal_matrix T>
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::diagonal> {};

  } // namespace detail


  /**
   * Derive the TriangleType from the type of the triangular_matrix.
   */
#ifdef __cpp_concepts
  template<triangular_matrix T>
  inline constexpr TriangleType triangle_type_of = detail::triangle_type_of_impl<T>::value;
#else
  template<typename T>
  inline constexpr TriangleType triangle_type_of = detail::triangle_type_of_impl<T>::value;
#endif


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether an object has elements that can be retrieved with N indices.
     * This should be specialized for all matrix types usable with OpenKalman.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N>
    struct is_element_gettable : std::false_type {};
#else
    template<typename T, std::size_t N, typename Enable = void>
    struct is_element_gettable : std::false_type {};
#endif
  }


  /**
   * T has elements that can be retrieved with N indices.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#else
  inline constexpr bool element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#endif


  namespace internal
  {
    /**
     * \internal A typed matrix or covariance T is gettable with N indices if its base matrix is likewise gettable.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N> requires (typed_matrix<T> or covariance<T>) and
      element_gettable<typename MatrixTraits<T>::BaseMatrix, N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<typed_matrix<T> or covariance<T>>>
      : is_element_gettable<typename MatrixTraits<T>::BaseMatrix, N> {};
#endif
  }


  // --------------------- //
  //  is_element_settable  //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether an object has elements that can be set with N indices.
     * This should be specialized for all matrix types usable with OpenKalman.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N>
    struct is_element_settable : std::false_type {};
#else
    template<typename T, std::size_t N, typename Enable = void>
    struct is_element_settable : std::false_type {};
#endif
  }

  /**
   * T has elements that can be set with N indices.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_settable = internal::is_element_settable<std::decay_t<T>, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#else
  inline constexpr bool element_settable = internal::is_element_settable<std::decay_t<T>, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#endif


  namespace internal
  {
    /**
     * \internal A typed matrix or covariance T is settable with N indices if its base matrix is likewise settable.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N> requires (typed_matrix<T> or covariance<T>) and
      element_settable<typename MatrixTraits<T>::BaseMatrix, N>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<typed_matrix<T> or covariance<T>>>
      : is_element_settable<typename MatrixTraits<T>::BaseMatrix, N> {};
#endif
  }

}

#endif //OPENKALMAN_TRAITS_HPP
