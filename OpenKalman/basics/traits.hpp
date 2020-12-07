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
 * \file
 * \brief Declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // =================================================================== //
  //  Traits that are fully defined independent of the matrix interface  //
  // =================================================================== //

  // ----------------- //
  //  nested_matrix_t  //
  // ----------------- //

  /**
   * \brief An alias for a type's nested matrix, if it exists.
   * \details Only participates in overload resolution if the type has a nested matrix.
   * \tparam T A type that is a wrapper for a nested matrix.
   */
#ifdef __cpp_concepts
  template<typename T> requires (requires {typename MatrixTraits<T>::NestedMatrix;})
#else
  template<typename T, typename = typename MatrixTraits<T>::NestedMatrix>
#endif
  using nested_matrix_t = typename MatrixTraits<T>::NestedMatrix;


  // ----------------- //
  //  native_matrix_t  //
  // ----------------- //

  /**
   * \brief An alias for a self-contained native matrix, based on and equivalent to parameter T.
   * \tparam T The type from which the native matrix is derived.
   * \tparam rows Number of rows in the native matrix (defaults to the number of rows in T).
   * \tparam cols Number of columns in the native matrix (defaults to the number of columns in T).
   */
  template<typename T, std::size_t rows = MatrixTraits<T>::dimension, std::size_t cols = MatrixTraits<T>::columns>
  using native_matrix_t = typename MatrixTraits<T>::template NativeMatrix<rows, cols>;


  /**
   * \brief Make a self-contained, native matrix based on the shape of M from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<typename M, std::convertible_to<typename MatrixTraits<M>::Scalar> ... Args> requires
    (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns)
#else
  template<typename M, typename ... Args, std::enable_if_t<
    (std::is_convertible_v<Args, typename MatrixTraits<M>::Scalar> and ...) and
    (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    return MatrixTraits<native_matrix_t<M>>::make(args...);
  }


  // ========================================================================== //
  //  Traits for which specializations must be defined in the matrix interface  //
  // ========================================================================== //

  // ---------------------------------- //
  //  contains_nested_lvalue_reference  //
  // ---------------------------------- //

  namespace internal
  {
#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename Enable = void>
      struct is_reference_dependent : std::false_type {};

      template<typename T>
      struct is_reference_dependent<T, std::enable_if_t<std::is_lvalue_reference_v<nested_matrix_t<T>>>>
        : std::true_type{};
    }
#endif


    /**
     * \internal
     * \brief Specifies that a matrix type contains an internal reference.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept contains_nested_lvalue_reference = std::is_reference_v<nested_matrix_t<T>>;
#else
    template<typename T>
    inline constexpr bool contains_nested_lvalue_reference = detail::is_reference_dependent<T>::value;
#endif

  } // internal


  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * \details Type trait testing whether T is self-contained (i.e., can be the return value of a function).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_self_contained : std::false_type {};
  }


  /**
   * \brief Specifies that a type is a self-contained matrix or expression.
   * \details A value is self-contained if it can be created in a function and returned as the result.
   * OpenKalman matrix types are self-contained if their wrapped native matrix is self-contained and is
   * not an lvalue reference.
   * The matrix library interface will specify which native matrices and expressions are self-contained.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_contained = internal::is_self_contained<std::decay_t<T>>::value;
#else
  inline constexpr bool self_contained = internal::is_self_contained<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /*
     * A typed matrix or covariance is self-contained if its nested matrix is self-contained and not an lvalue ref.
     */
#ifdef __cpp_concepts
    template<typename T> requires (typed_matrix<T> or covariance<T>) and self_contained<nested_matrix_t<T>> and
      (not internal::contains_nested_lvalue_reference<T>)
    struct is_self_contained<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<(typed_matrix<T> or covariance<T>) and
      (not internal::contains_nested_lvalue_reference<T>)>> : is_self_contained<nested_matrix_t<T>> {};
#endif


    /*
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


  // ---------------- //
  //  is_zero_matrix  //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a zero matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_zero_matrix : std::false_type {};
  }


  /**
   * \brief Specifies that a type is a zero matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /*
     * A typed matrix or covariance is a zero matrix if its nested matrix is a zero matrix.
     */
#ifdef __cpp_concepts
    template<typename T> requires (typed_matrix<T> or covariance<T>) and zero_matrix<nested_matrix_t<T>>
    struct is_zero_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<(typed_matrix<T> or covariance<T>)>>
      : is_zero_matrix<nested_matrix_t<T>> {};
#endif


    /*
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
     * \brief A type trait testing whether an object is an identity matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_identity_matrix : std::false_type {};
  }

  /**
   * \brief Specifies that a type is an identity matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /*
     * A covariance is an identity matrix if its nested matrix is an identity matrix.
     */
#ifdef __cpp_concepts
    template<covariance T> requires identity_matrix<nested_matrix_t<T>>
    struct is_identity_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<covariance<T>>> : is_identity_matrix<nested_matrix_t<T>> {};
#endif


    /*
     * A typed matrix is an identity matrix if its nested matrix is an identity matrix
     * and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires identity_matrix<nested_matrix_t<T>> and
    equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_identity_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_identity_matrix<nested_matrix_t<T>> {};
#endif
  }


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a diagonal matrix
     * \note This excludes zero_matrix, identity_matrix, or one_by_one_matrix.
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
   * \brief Specifies that a type is a diagonal matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
    /*
     * A covariance is a diagonal matrix if its nested matrix is diagonal
     */
#ifdef __cpp_concepts
    template<covariance T> requires diagonal_matrix<nested_matrix_t<T>>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<covariance<T>>> : is_diagonal_matrix<nested_matrix_t<T>> {};
#endif


    /*
     * A typed matrix is diagonal if its nested matrix is diagonal and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires diagonal_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_diagonal_matrix<nested_matrix_t<T>> {};
#endif


    /*
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
     * \brief A type trait testing whether an object is a self-adjoint matrix (other than diagonal_matrix).
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
   * \brief Specifies that a type is a self-adjoint matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
    /*
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


    /*
     * A typed matrix is self-adjoint if its nested matrix is self-adjoint and
     * its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires self_adjoint_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_self_adjoint_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : is_self_adjoint_matrix<nested_matrix_t<T>> {};
#endif


    /*
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
     * \brief A type trait testing whether an object is a lower-triangular matrix (other than diagonal_matrix).
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
   * \brief Specifies that a type is a lower-triangular matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
    /*
     * A square root covariance is lower-triangular if its nested matrix is either lower-triangular or self-adjoint.
     * If the nested matrix is self-adjoint, the square root covariance can be either lower- or upper-triangular.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires lower_triangular_matrix<nested_matrix_t<T>> or
      self_adjoint_matrix<nested_matrix_t<T>>
    struct is_lower_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
      : std::bool_constant<lower_triangular_matrix<nested_matrix_t<T>> or
          self_adjoint_matrix<nested_matrix_t<T>>> {};
#endif
  }


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is an upper-triangular matrix (other than diagonal_matrix).
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
   * \brief Specifies that a type is an upper-triangular matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
    /*
     * A square root covariance is upper-triangular if its nested matrix is either upper-triangular or self-adjoint.
     * If the nested matrix is self-adjoint, the square root covariance can be either lower- or upper-triangular.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires upper_triangular_matrix<nested_matrix_t<T>> or
      self_adjoint_matrix<nested_matrix_t<T>>
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
        : std::bool_constant<upper_triangular_matrix<nested_matrix_t<T>> or
            self_adjoint_matrix<nested_matrix_t<T>>> {};
#endif
  }


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether an object has elements that can be retrieved with N indices.
     * \note This should be specialized for all matrix types usable with OpenKalman.
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
   * \brief Specifies that a type has elements that can be retrieved with N number of indices.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#else
  inline constexpr bool element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#endif


  namespace internal
  {
    /*
     * A typed matrix or covariance T is gettable with N indices if its nested matrix is likewise gettable.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N> requires (typed_matrix<T> or covariance<T>) and
      element_gettable<nested_matrix_t<T>, N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<typed_matrix<T> or covariance<T>>>
      : is_element_gettable<nested_matrix_t<T>, N> {};
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
     * \note This should be specialized for all matrix types usable with OpenKalman.
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
   * \brief Specifies that a type has elements that can be set with N number of indices.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_settable = internal::is_element_settable<T, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#else
  inline constexpr bool element_settable = internal::is_element_settable<T, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#endif


  namespace internal
  {
    /*
     * A typed matrix or covariance T is settable with N indices if its nested matrix is likewise settable.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N> requires (typed_matrix<T> or covariance<T>) and
      element_settable<nested_matrix_t<T>, N>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<(typed_matrix<T> or covariance<T>) and
      element_settable<nested_matrix_t<T>, N>>>
      : std::true_type {};
#endif
  }


  // ======================================================================================= //
  //  Traits that are fully defined here, but depend on definitions in the matrix interface  //
  // ======================================================================================= //

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
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   */
  template<typename T>
  using self_contained_t = std::conditional_t<
    self_contained<T>, std::decay_t<T>, typename detail::self_contained_impl<T>::type>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \details A passable type is either an lvalue reference or an rvalue reference to a self_contained_t type.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, self_contained_t<T>>;


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
   * \brief Specifies that a type is a triangular matrix (upper or lower).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
     * \internal
     * \brief Specifies that two types have the same triangular type (upper or lower).
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
     * \brief A type trait testing whether the nested matrix is a Cholesky square root.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_cholesky_form : std::false_type {};
  }


  /**
   * \brief Specifies that a type has a nested native matrix that is a Cholesky square root.
   * \details If this is true, then nested_matrix_t<T> is true.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#else
  inline constexpr bool cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /*
     * A covariance has a Cholesky form if its nested matrix is not self-adjoint.
     */
#ifdef __cpp_concepts
    template<covariance T> requires (not self_adjoint_matrix<nested_matrix_t<T>>)
    struct is_cholesky_form<T> : std::true_type {};
#else
    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<covariance<T>>>
      : std::bool_constant<not self_adjoint_matrix<nested_matrix_t<T>>> {};
#endif


    /*
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
  } // namespace internal


  // -------------------------------- //
  //  TriangleType, triangle_type_of  //
  // -------------------------------- //

  /**
   * \brief The type of a triangular matrix, either lower, upper, or diagonal.
   */
  enum struct TriangleType {
    lower, ///< The lower-left triangle.
    upper, ///< The upper-right triangle.
    diagonal ///< The diagonal elements of the matrix.
    };


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
   * \brief Derive the TriangleType from the type of the triangular_matrix.
   */
#ifdef __cpp_concepts
  template<triangular_matrix T>
#else
  template<typename T>
#endif
  inline constexpr TriangleType triangle_type_of = detail::triangle_type_of_impl<T>::value;

}

#endif //OPENKALMAN_TRAITS_HPP
