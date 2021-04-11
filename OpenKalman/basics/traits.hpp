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

    /*
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


    /*
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


    // ---------------- //
    //  is_zero_matrix  //
    // ---------------- //

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


    // -------------------- //
    //  is_identity_matrix  //
    // -------------------- //

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


    // -------------------- //
    //  is_square_matrix  //
    // -------------------- //

#ifdef __cpp_concepts
    template<typed_matrix T> requires (MatrixTraits<T>::rows == MatrixTraits<T>::columns) and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_square_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_square_matrix<T, std::enable_if_t<typed_matrix<T> and
      (MatrixTraits<T>::rows == MatrixTraits<T>::columns) and
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


#ifdef __cpp_concepts
    template<typename T> requires (not typed_matrix<T>) and (not covariance<T>) and
      (MatrixTraits<T>::rows == MatrixTraits<T>::columns)
    struct is_square_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_square_matrix<T, std::enable_if_t<(not typed_matrix<T>) and (not covariance<T>) and
      (MatrixTraits<T>::rows == MatrixTraits<T>::columns)>> : std::true_type {};
#endif


    // -------------------- //
    //  is_diagonal_matrix  //
    // -------------------- //

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


    // ------------------------ //
    //  is_self_adjoint_matrix  //
    // ------------------------ //

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


    // ---------------------------- //
    //  is_lower_triangular_matrix  //
    // ---------------------------- //

    /*
     * A square root covariance is lower-triangular based on its MatrixTraits.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires (MatrixTraits<T>::triangle_type != TriangleType::upper)
    struct is_lower_triangular_matrix<T> : std::true_type {};
#else
  template<typename T>
  struct is_lower_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
  : std::bool_constant<MatrixTraits<T>::triangle_type != TriangleType::upper> {};
#endif


    /*
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

    /*
     * A square root covariance is upper-triangular based on its MatrixTraits.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T> requires (MatrixTraits<T>::triangle_type != TriangleType::lower)
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<square_root_covariance<T>>>
      : std::bool_constant<MatrixTraits<T>::triangle_type != TriangleType::lower> {};
#endif


    /*
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
    //  is_cholesky_form  //
    // ------------------ //

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


    // --------------------- //
    //  is_element_gettable  //
    // --------------------- //

    /*
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


    /*
     * A non-square-root \ref covariance T is gettable with N indices if its self-adjoint nested matrix
     * is likewise gettable.
     */
#ifdef __cpp_concepts
    template<covariance T, std::size_t N> requires (not square_root_covariance<T>) and
      element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<covariance<T> and (not square_root_covariance<T>) and
      element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), N>>>
      : std::true_type {};
#endif

    /*
     * A \ref square_root_covariance T is gettable with N indices if its triangular nested matrix is likewise gettable.
     */
#ifdef __cpp_concepts
    template<square_root_covariance T, std::size_t N> requires
    element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<square_root_covariance<T> and
      element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), N>>>
      : std::true_type {};
#endif


    // --------------------- //
    //  is_element_settable  //
    // --------------------- //

    /*
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


    /*
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
    private:

      using non_const_type = typename MatrixTraits<T>::SelfContainedFrom;

      static_assert(not zero_matrix<T> or zero_matrix<non_const_type>,
        "If T is a zero_matrix, then MatrixTraits<T>::SelfContainedFrom must also be a zero_matrix.");
      static_assert(not identity_matrix<T> or identity_matrix<non_const_type>,
        "If T is an identity_matrix, then MatrixTraits<T>::SelfContainedFrom must also be an identity_matrix.");
      static_assert(not upper_triangular_matrix<T> or upper_triangular_matrix<non_const_type>, "If T is an "
        "upper_triangular_matrix, then MatrixTraits<T>::SelfContainedFrom must also be an upper_triangular_matrix.");
      static_assert(not lower_triangular_matrix<T> or lower_triangular_matrix<non_const_type>, "If T is a "
        "lower_triangular_matrix, then MatrixTraits<T>::SelfContainedFrom must also be a lower_triangular_matrix.");
      static_assert(not self_adjoint_matrix<T> or self_adjoint_matrix<non_const_type>,
        "If T is a self_adjoint_matrix, then MatrixTraits<T>::SelfContainedFrom must also be a self_adjoint_matrix.");

    public:

      using type = std::conditional_t<std::is_const_v<T>, const non_const_type, non_const_type>;

    };


#ifdef __cpp_concepts
    template<distribution T> requires (not self_contained<T>)
    struct self_contained_impl<T>
#else
      template<typename T>
    struct self_contained_impl<T, std::enable_if_t<distribution<T> and (not self_contained<T>)>>
#endif
    {
      using type = typename DistributionTraits<T>::SelfContainedFrom;

    private:
      static_assert(self_contained<typename DistributionTraits<T>::Mean>);
      static_assert(self_contained<typename DistributionTraits<T>::Covariance>);
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
   * \tparam Ts Other types (optional) that must also be lvalue references if T is not self-contained.
   * \details A passable type T is either an lvalue reference (and all other Ts are also lvalue references) or
   * is \ref self_contained_t.
   */
  template<typename T, typename...Ts>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, self_contained_t<T>>;


  // ====================== //
  //  Constant expressions  //
  // ====================== //

  // ------------------ //
  //  triangle_type_of  //
  // ------------------ //

  namespace detail
  {
#ifdef __cpp_concepts
    template<triangular_matrix T>
#else
    template<typename T, typename = void>
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


  // =========== //
  //  Functions  //
  // =========== //

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
    constexpr decltype(auto) to_covariance_nestable(Arg&&) noexcept;


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
     * /return A \ref triangular_matrix if Arg is a \ref square_root_covariance or otherwise a \ref self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto) to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


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
    template<typename T, typename U> requires (MatrixTraits<T>::rows == MatrixTraits<U>::rows) and
      (MatrixTraits<T>::columns == MatrixTraits<U>::columns) and
      (std::same_as<typename MatrixTraits<T>::Scalar, typename MatrixTraits<U>::Scalar>)
    struct has_same_matrix_shape<T, U> : std::true_type {};
#else
    template<typename T, typename U>
    struct has_same_matrix_shape<T, U, std::enable_if_t<
      (MatrixTraits<T>::rows == MatrixTraits<U>::rows) and
      (MatrixTraits<T>::columns == MatrixTraits<U>::columns) and
      (std::is_same_v<typename MatrixTraits<T>::Scalar, typename MatrixTraits<U>::Scalar>)>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not has_same_matrix_shape<T, U>::value) or
      (zero_matrix<T> and not zero_matrix<U>) or
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
      (zero_matrix<T> and not zero_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (upper_triangular_matrix<T> and not upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and not lower_triangular_matrix<U>) or
      (self_adjoint_matrix<T> and not self_adjoint_matrix<U>)>> : std::false_type {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_TRAITS_HPP
