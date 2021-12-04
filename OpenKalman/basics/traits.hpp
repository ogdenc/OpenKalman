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
  // ========================================================================== //
  //  Traits for which specializations must be defined in the matrix interface  //
  // ========================================================================== //

  namespace internal
  {

    // ------------------- //
    //  is_self_contained  //
    // ------------------- //

    /**
     * A typed matrix or covariance is self-contained if its nested matrix is self-contained and not an lvalue ref.
     */
#ifdef __cpp_concepts
    template<typename T> requires (typed_matrix<T> or covariance<T>) and self_contained<nested_matrix_t<T>>
    struct is_self_contained<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
      : is_self_contained<nested_matrix_t<T>> {};
#endif


    /**
     * A distribution is self-contained if its associated mean and covariance are also self-contained.
     */
#ifdef __cpp_concepts
    template<distribution T>
    requires self_contained<typename DistributionTraits<T>::Mean> and
      self_contained<typename DistributionTraits<T>::Covariance>
    struct is_self_contained<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<distribution<T> and
      self_contained<typename DistributionTraits<T>::Mean> and
      self_contained<typename DistributionTraits<T>::Covariance>>>
      : std::true_type {};
#endif

  } // namespace internal


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
    : constant_diagonal_coefficient<std::decay_t<nested_matrix_t<T>>> {};


    // ---------------------- //
    //  constant_coefficient  //
    // ---------------------- //

  /**
   * An constant_diagonal_coefficient is a constant matrix if it is one-by-one.
   */
#ifdef __cpp_concepts
  template<constant_diagonal_matrix T> requires (not typed_matrix<T>) and (not covariance<T>) and one_by_one_matrix<T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<(not typed_matrix<T>) and (not covariance<T>) and
    constant_diagonal_matrix<T> and one_by_one_matrix<T>>>
#endif
    : constant_diagonal_coefficient<T> {};


  /**
   * An constant_diagonal_coefficient is a constant matrix if it is zero.
   */
#ifdef __cpp_concepts
  template<constant_diagonal_matrix T> requires (not typed_matrix<T>) and (not covariance<T>) and
    (constant_diagonal_coefficient_v<T> == 0)
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<(not typed_matrix<T>) and (not covariance<T>) and
    constant_diagonal_matrix<T> and (constant_diagonal_coefficient_v<T> == 0)>>
#endif
    : internal::constant_coefficient_type<short {0}, typename MatrixTraits<T>::Scalar> {};


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
      (triangular_covariance<T> and zero_matrix<nested_matrix_t<T>>)>>
#endif
    : constant_coefficient<std::decay_t<nested_matrix_t<T>>> {};


  namespace internal
  {
    // -------------------- //
    //  is_diagonal_matrix  //
    // -------------------- //

    /**
     * A covariance is a diagonal matrix if its nested matrix is diagonal
     */
#ifdef __cpp_concepts
    template<covariance T> requires diagonal_matrix<nested_matrix_t<T>>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<covariance<T> and diagonal_matrix<nested_matrix_t<T>>>>
      : std::true_type {};
#endif


    /**
     * A typed matrix is diagonal if its nested matrix is diagonal and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires diagonal_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<typed_matrix<T> and diagonal_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : std::true_type {};
#endif


    // ------------------------------ //
    //  is_lower_self_adjoint_matrix  //
    // ------------------------------ //

    /**
     * A self_adjoint_covariance is lower-self-adjoint if it its covariance is lower-triangular or lower-self-adjoint.
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance T>
    requires lower_self_adjoint_matrix<nested_matrix_t<T>> or lower_triangular_matrix<nested_matrix_t<T>>
    struct is_lower_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_lower_self_adjoint_matrix<T, std::enable_if_t<self_adjoint_covariance<T> and
      lower_self_adjoint_matrix<nested_matrix_t<T>> or lower_triangular_matrix<nested_matrix_t<T>>>>
#endif
      : std::true_type {};


    /**
     * A typed matrix is lower-self-adjoint if its nested matrix is lower-self-adjoint and
     * its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires lower_self_adjoint_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_lower_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_lower_self_adjoint_matrix<T, std::enable_if_t<
      typed_matrix<T> and lower_self_adjoint_matrix<nested_matrix_t<T>> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
#endif
      : std::true_type {};


  // ------------------------------ //
  //  is_upper_self_adjoint_matrix  //
  // ------------------------------ //

  /**
   * A self_adjoint_covariance is upper-self-adjoint if it its covariance is upper-triangular or upper-self-adjoint.
   */
#ifdef __cpp_concepts
  template<self_adjoint_covariance T>
  requires upper_self_adjoint_matrix<nested_matrix_t<T>> or upper_triangular_matrix<nested_matrix_t<T>>
  struct is_upper_self_adjoint_matrix<T>
#else
  template<typename T>
  struct is_upper_self_adjoint_matrix<T, std::enable_if_t<self_adjoint_covariance<T> and
    upper_self_adjoint_matrix<nested_matrix_t<T>> or upper_triangular_matrix<nested_matrix_t<T>>>>
#endif
    : std::true_type {};


  /**
   * A typed matrix is upper-self-adjoint if its nested matrix is upper-self-adjoint and
   * its row and column coefficients are equivalent.
   */
#ifdef __cpp_concepts
  template<typed_matrix T> requires upper_self_adjoint_matrix<nested_matrix_t<T>> and
    equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
  struct is_upper_self_adjoint_matrix<T>
#else
  template<typename T>
  struct is_upper_self_adjoint_matrix<T, std::enable_if_t<
    typed_matrix<T> and upper_self_adjoint_matrix<nested_matrix_t<T>> and
    equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
#endif
    : std::true_type {};


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
      : std::bool_constant<lower_triangular_matrix<std::decay_t<nested_matrix_t<T>>> or
        lower_self_adjoint_matrix<std::decay_t<nested_matrix_t<T>>>> {};


    /**
     * A typed matrix is lower_triangular if its nested matrix is lower_triangular.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires lower_triangular_matrix<nested_matrix_t<T>>
    struct is_lower_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<typed_matrix<T>>>
      : is_lower_triangular_matrix<nested_matrix_t<T>> {};
#endif


    // ---------------------------- //
    //  is_upper_triangular_matrix  //
    // ---------------------------- //

    /**
     * A triangular_covariance is upper-triangular based on its triangle type.
     */
#ifdef __cpp_concepts
    template<triangular_covariance T> requires (triangle_type_of_v<T> != TriangleType::lower)
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<triangular_covariance<T>>>
      : std::bool_constant<triangle_type_of_v<T> != TriangleType::lower> {};
#endif


    /**
     * A typed matrix is upper_triangular if its nested matrix is upper_triangular.
     */
#ifdef __cpp_concepts
    template<typed_matrix T> requires upper_triangular_matrix<nested_matrix_t<T>>
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<typed_matrix<T>>>
      : is_upper_triangular_matrix<nested_matrix_t<T>> {};
#endif


    // ------------------ //
    //  is_square_matrix  //
    // ------------------ //

#ifdef __cpp_concepts
    template<typed_matrix T> requires (not dynamic_shape<T>) and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_square_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_square_matrix<T, std::enable_if_t<typed_matrix<T> and (not dynamic_shape<T>) and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
      : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<covariance T>
    struct is_square_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_square_matrix<T, std::enable_if_t<covariance<T>>>
      : std::true_type {};
#endif


    // --------------------- //
    //  is_element_gettable  //
    // --------------------- //

    /**
     * A typed matrix is gettable with N indices if its nested matrix is likewise gettable.
     */
#ifdef __cpp_concepts
    template<typed_matrix T, std::size_t N> requires element_gettable<nested_matrix_t<T>, N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<typed_matrix<T> and element_gettable<nested_matrix_t<T>, N>>>
      : std::true_type {};
#endif


    /**
     * A non-square-root \ref covariance T is gettable with N indices if its self-adjoint nested matrix
     * is likewise gettable.
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance T, std::size_t N>
    requires element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<self_adjoint_covariance<T> and
      element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), N>>>
      : std::true_type {};
#endif

    /**
     * A \ref triangular_covariance T is gettable with N indices if its triangular nested matrix is likewise gettable.
     */
#ifdef __cpp_concepts
    template<triangular_covariance T, std::size_t N> requires
    element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<triangular_covariance<T> and
      element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), N>>>
      : std::true_type {};
#endif


    // --------------------- //
    //  is_element_settable  //
    // --------------------- //

    /**
     * A typed matrix is settable with N indices if its nested matrix is likewise settable.
     */
#ifdef __cpp_concepts
    template<typed_matrix T, std::size_t N> requires element_settable<nested_matrix_t<T>, N>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<typed_matrix<T> and element_settable<nested_matrix_t<T>, N>>>
      : std::true_type {};
#endif


    /**
     * A covariance T is settable with N indices if its nested matrix is likewise settable.
     */
#ifdef __cpp_concepts
    template<covariance T, std::size_t N> requires element_settable<nested_matrix_t<T>, N>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<covariance<T> and element_settable<nested_matrix_t<T>, N>>>
      : std::true_type {};
#endif


  } // namespace internal


  // ========= //
  //  Aliases  //
  // ========= //

  // ----------------- //
  //  native_matrix_t  //
  // ----------------- //

  /**
   * \brief An alias for a self-contained native matrix, based on and equivalent to parameter T.
   * \tparam T The type from which the native matrix is derived.
   * \tparam rows Number of rows in the native matrix (defaults to the number of rows in T).
   * \tparam cols Number of columns in the native matrix (defaults to the number of columns in T).
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   */
  template<typename T,
    std::size_t rows = dynamic_rows<T> and square_matrix<T> and not dynamic_columns<T> ?
      MatrixTraits<T>::columns : MatrixTraits<T>::rows,
    std::size_t cols = dynamic_columns<T> and square_matrix<T> and not dynamic_rows<T> ?
      MatrixTraits<T>::rows : MatrixTraits<T>::columns,
    typename Scalar = typename MatrixTraits<T>::Scalar>
  using native_matrix_t = typename MatrixTraits<T>::template NativeMatrixFrom<rows, cols, Scalar>;


  // ------------------ //
  //  self_contained_t  //
  // ------------------ //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct self_contained_impl { using type = T; };


#ifdef __cpp_concepts
    template<typename T> requires (not self_contained<T>) and (not distribution<T>)
    struct self_contained_impl<T>
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<(not self_contained<T>) and (not distribution<T>)>>
#endif
    {
      using type = std::conditional_t<
        std::is_const_v<T>,
        const typename MatrixTraits<T>::SelfContainedFrom,
        typename MatrixTraits<T>::SelfContainedFrom>;
    };


#ifdef __cpp_concepts
    template<distribution T> requires (not self_contained<T>)
    struct self_contained_impl<T>
#else
      template<typename T>
    struct self_contained_impl<T, std::enable_if_t<distribution<T> and (not self_contained<T>)>>
#endif
    {
      using type = std::conditional_t<
        std::is_const_v<T>,
        const typename DistributionTraits<T>::SelfContainedFrom,
        typename DistributionTraits<T>::SelfContainedFrom>;
    };

  } // namespace detail


  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   */
  template<typename T>
  using self_contained_t = typename detail::self_contained_impl<std::remove_reference_t<T>>::type;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \tparam T The type in question.
   * \details A passable type T is either an lvalue reference or is \ref self_contained_t.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, self_contained_t<T>>;


  // =========== //
  //  Functions  //
  // =========== //

  // -------------------- //
  //  make_native_matrix  //
  // -------------------- //

#ifdef __cpp_concepts
  template<typename M, std::convertible_to<const typename MatrixTraits<M>::Scalar> ... Args>
  requires ((MatrixTraits<M>::rows == 0 and MatrixTraits<M>::columns == 0) or
    (MatrixTraits<M>::rows != 0 and
      MatrixTraits<M>::columns != 0 and sizeof...(Args) == MatrixTraits<M>::rows * MatrixTraits<M>::columns) or
    (MatrixTraits<M>::columns == 0 and sizeof...(Args) % MatrixTraits<M>::rows == 0) or
    (MatrixTraits<M>::rows == 0 and sizeof...(Args) % MatrixTraits<M>::columns == 0)) and
    requires { typename MatrixTraits<native_matrix_t<M>>; }
#else
  #pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, typename ... Args, std::enable_if_t<
    (std::is_convertible_v<Args, const typename MatrixTraits<M>::Scalar> and ...) and
    ((MatrixTraits<M>::rows == 0 and MatrixTraits<M>::columns == 0) or
    (MatrixTraits<M>::rows != 0 and
      MatrixTraits<M>::columns != 0 and sizeof...(Args) == MatrixTraits<M>::rows * MatrixTraits<M>::columns) or
    (MatrixTraits<M>::columns == 0 and sizeof...(Args) % MatrixTraits<M>::rows == 0) or
    (MatrixTraits<M>::rows == 0 and sizeof...(Args) % MatrixTraits<M>::columns == 0)), int> = 0,
    typename = std::void_t<MatrixTraits<native_matrix_t<M>>>>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Nat = native_matrix_t<M>;
    return MatrixTraits<Nat>::make(static_cast<const typename MatrixTraits<M>::Scalar>(args)...);
  }
#ifndef __cpp_concepts
#pragma GCC diagnostic pop
#endif


  namespace internal
  {
    // ------------------------ //
    //  to_covariance_nestable  //
    // ------------------------ //

    /**
     * \overload
     * \internal
     * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \internal
     * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \tparam Arg A \ref covariance or \ref typed_matrix.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (MatrixTraits<Arg>::rows == MatrixTraits<T>::rows) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return The result of converting Arg to a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
        (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


  // ------------- //
  //  is_writable  //
  // ------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typed_matrix T> requires writable<nested_matrix_t<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<typed_matrix<T> and writable<nested_matrix_t<T>>>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<covariance T> requires writable<nested_matrix_t<T>>
    struct is_writable<T> : std::true_type {};
#else
    template<typename T>
    struct is_writable<T, std::enable_if_t<covariance<T> and writable<nested_matrix_t<T>>>> : std::true_type {};
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
      (requires { typename nested_matrix_t<T>; } and has_const<nested_matrix_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct has_same_matrix_shape : std::false_type {};


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      (dynamic_rows<T> or dynamic_rows<U> or MatrixTraits<T>::rows == MatrixTraits<U>::rows) and
      (dynamic_columns<T> or dynamic_columns<U> or MatrixTraits<T>::columns == MatrixTraits<U>::columns) and
      (std::same_as<typename MatrixTraits<T>::Scalar, typename MatrixTraits<U>::Scalar>)
    struct has_same_matrix_shape<T, U> : std::true_type {};
#else
    template<typename T, typename U>
    struct has_same_matrix_shape<T, U, std::enable_if_t<
      (dynamic_rows<T> or dynamic_rows<U> or MatrixTraits<T>::rows == MatrixTraits<U>::rows) and
      (dynamic_columns<T> or dynamic_columns<U> or MatrixTraits<T>::columns == MatrixTraits<U>::columns) and
      (std::is_same_v<typename MatrixTraits<T>::Scalar, typename MatrixTraits<U>::Scalar>)>> : std::true_type {};
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
