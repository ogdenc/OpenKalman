/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  namespace Eigen3
  {
    // ---------------------------- //
    //    New Eigen matrix types    //
    // ---------------------------- //

    /**
     * A self-adjoint matrix, based on an Eigen matrix.
     * @tparam BaseMatrix The Eigen matrix on which the self-adjoint matrix is based.
     * @tparam storage_triangle The triangle (TriangleType::upper or TriangleType::lower) in which the data is stored.
     */
    template<typename BaseMatrix, TriangleType storage_triangle = TriangleType::lower>
    struct SelfAdjointMatrix;

    /**
     * A triangular matrix, based on an Eigen matrix.
     * @tparam BaseMatrix The Eigen matrix on which the triangular matrix is based.
     * @tparam triangle_type The triangle (TriangleType::upper or TriangleType::lower).
     */
    template<typename BaseMatrix, TriangleType triangle_type = TriangleType::lower>
    struct TriangularMatrix;

    /**
     * A diagonal matrix, based on an Eigen matrix.
     *
     * Note: This has the same name as Eigen::DiagonalMatrix, and is intended as an improved replacement.
     * @tparam BaseMatrix A single-column matrix defining the diagonal.
     */
    template<typename BaseMatrix>
    struct DiagonalMatrix;

    /**
     * A wrapper type for an Eigen zero matrix.
     *
     * This is necessary because Eigen3 types do not distinguish between a zero matrix and a constant matrix.
     * @tparam BaseMatrix The Eigen matrix type on which the zero matrix is based. Only its shape is relevant.
     */
    template<typename BaseMatrix>
    struct ZeroMatrix;

    /**
     * An expression that transforms angular or other modular coefficients into Euclidean space, for proper wrapping.
     *
     * This is the counterpart expression to ToEuclideanExpr.
     * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in <code>B</code>.
     * @tparam Coefficients The coefficient types.
     * @tparam BaseMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
     */
    template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
    struct ToEuclideanExpr;

    /**
     * An expression that transforms angular or other modular coefficients back from Euclidean space.
     *
     * This is the counterpart expression to ToEuclideanExpr.
     * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in <code>B</code>.
     * @tparam Coefficients The coefficient types.
     * @tparam BaseMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
     */
    template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
    struct FromEuclideanExpr;


    // ------------------------------------------------------- //
    //    Concepts / type traits for new Eigen matrix types    //
    // ------------------------------------------------------- //

#ifdef __cpp_concepts
    /**
     * Type T is a self-adjoint matrix based on the Eigen library.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
    template<typename T>
    concept eigen_self_adjoint_expr = std::same_as<
      std::decay_t<T>, SelfAdjointMatrix<typename MatrixTraits<T>::BaseMatrix, MatrixTraits<T>::storage_type>>;
#else
    /**
     * Tests whether type T is a self-adjoint matrix based on the Eigen library.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
    template<typename T>
    struct is_eigen_self_adjoint_expr : internal::class_trait<is_eigen_self_adjoint_expr, T> {};

    template<typename T>
    inline constexpr bool is_eigen_self_adjoint_expr_v = is_eigen_self_adjoint_expr<T>::value;

    template<typename T>
    inline constexpr bool eigen_self_adjoint_expr = is_eigen_self_adjoint_expr_v<T>;

    template<typename BaseMatrix, TriangleType storage_triangle>
    struct is_eigen_self_adjoint_expr<SelfAdjointMatrix<BaseMatrix, storage_triangle>> : std::true_type {};
#endif

    /**
     * A triangular matrix based on the Eigen library.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept eigen_triangular_expr = std::same_as<
      std::decay_t<T>, TriangularMatrix<typename MatrixTraits<T>::BaseMatrix, MatrixTraits<T>::triangle_type>>;
#else
    template<typename T>
    struct is_eigen_triangular_expr : internal::class_trait<is_eigen_triangular_expr, T> {};

    template<typename T>
    inline constexpr bool is_eigen_triangular_expr_v = is_eigen_triangular_expr<T>::value;

    template<typename T>
    inline constexpr bool eigen_triangular_expr = is_eigen_triangular_expr_v<T>;

    template<typename BaseMatrix, TriangleType triangle_type>
    struct is_eigen_triangular_expr<TriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};
#endif

    /**
     * A diagonal matrix based on the Eigen library.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept eigen_diagonal_expr = std::same_as<std::decay_t<T>, DiagonalMatrix<typename MatrixTraits<T>::BaseMatrix>>;
#else
    template<typename T>
    struct is_eigen_diagonal_expr : internal::class_trait<is_eigen_diagonal_expr, T> {};

    template<typename T>
    inline constexpr bool is_eigen_diagonal_expr_v = is_eigen_diagonal_expr<T>::value;

    template<typename T>
    inline constexpr bool eigen_diagonal_expr = is_eigen_diagonal_expr_v<T>;

    template<typename BaseMatrix>
    struct is_eigen_diagonal_expr<DiagonalMatrix<BaseMatrix>> : std::true_type {};
#endif

    /**
     * A zero matrix based on the Eigen library. (All coefficients are zero.)
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept eigen_zero_expr = std::same_as<std::decay_t<T>, ZeroMatrix<typename MatrixTraits<T>::BaseMatrix>>;
#else
    template<typename T>
    struct is_eigen_zero_expr : internal::class_trait<is_eigen_zero_expr, T> {};

    template<typename T>
    inline constexpr bool is_eigen_zero_expr_v = is_eigen_zero_expr<T>::value;

    template<typename T>
    inline constexpr bool eigen_zero_expr = is_eigen_zero_expr_v<T>;

    template<typename BaseMatrix>
    struct is_eigen_zero_expr<ZeroMatrix<BaseMatrix>> : std::true_type {};
#endif

    /**
     * An expression converting each column vector in a Euclidean Eigen matrix from Euclidean space.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept from_euclidean_expr = std::same_as<std::decay_t<T>,
      FromEuclideanExpr<typename MatrixTraits<T>::Coefficients, typename MatrixTraits<T>::BaseMatrix>>;
#else
    template<typename T>
    struct is_from_euclidean_expr : internal::class_trait<is_from_euclidean_expr, T> {};

    template<typename T>
    inline constexpr bool is_from_euclidean_expr_v = is_from_euclidean_expr<T>::value;

    template<typename T>
    inline constexpr bool from_euclidean_expr = is_from_euclidean_expr_v<T>;

    template<typename Coefficients, typename BaseMatrix>
    struct is_from_euclidean_expr<FromEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};
#endif


    /**
     * An expression converting each column vector in an Eigen matrix to Euclidean space.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept to_euclidean_expr = std::same_as<std::decay_t<T>,
      ToEuclideanExpr<typename MatrixTraits<T>::Coefficients, typename MatrixTraits<T>::BaseMatrix>>;
#else
    template<typename T>
    struct is_to_euclidean_expr : internal::class_trait<is_to_euclidean_expr, T> {};

    template<typename T>
    inline constexpr bool is_to_euclidean_expr_v = is_to_euclidean_expr<T>::value;

    template<typename T>
    inline constexpr bool to_euclidean_expr = is_to_euclidean_expr_v<T>;

    template<typename Coefficients, typename BaseMatrix>
    struct is_to_euclidean_expr<ToEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};
#endif


    // --------------------------------- //
    //    General concepts and traits    //
    // --------------------------------- //

    /**
     * Either from_euclidean_expr or to_euclidean_expr.
     * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
     */
#ifdef __cpp_concepts
    template<typename T>
    concept euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#else
    template<typename T>
    struct is_euclidean_expr : std::bool_constant<is_from_euclidean_expr_v<T> or is_to_euclidean_expr_v<T>> {};

    /// Helper template for is_eigen_native.
    template<typename T>
    inline constexpr bool euclidean_expr = is_euclidean_expr<T>::value;
#endif

    namespace internal
    {
      /*
       * A non-native Eigen type used in the Eigen interface for OpenKalman.
       * For compatibility, this is also defined as an inline constexpr bool variable in c++17.
       */
#ifdef __cpp_concepts
      template<typename T>
      concept eigen_new =
      eigen_self_adjoint_expr<T> or
        eigen_triangular_expr<T> or
        eigen_diagonal_expr<T> or
        eigen_zero_expr<T> or
        euclidean_expr<T>;
#else
      template<typename T>
      struct is_eigen_new : std::integral_constant<bool,
        is_eigen_self_adjoint_expr_v<T> or
          is_eigen_triangular_expr_v<T> or
          is_eigen_diagonal_expr_v<T> or
          is_eigen_zero_expr_v<T> or
          euclidean_expr<T>>
      {
      };

      /// Helper template for is_eigen_native.
      template<typename T>
      inline constexpr bool eigen_new = is_eigen_new<T>::value;
#endif
    }

#ifdef __cpp_concepts
    /// An object that is a native Eigen::MatrixBase type in Eigen3.
    template<typename T>
    concept eigen_native = requires {typename MatrixTraits<std::decay_t<T>>::BaseMatrix;} and
      (not internal::eigen_new<T>) and (not is_typed_matrix_v<T>) and (not is_covariance_v<T>);
#else
    /// Whether an object is a native Eigen::MatrixBase type in Eigen3.
    template<typename T>
    struct is_eigen_native : std::integral_constant<bool,
      std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
        not internal::eigen_new<T> and not is_typed_matrix_v < T> and not is_covariance_v <T>> {};

    /// Helper template for is_eigen_native.
    template<typename T>
    inline constexpr bool is_eigen_native_v = is_eigen_native<T>::value;

    /// Synonym for is_eigen_native_v.
    template<typename T>
    inline constexpr bool eigen_native = is_eigen_native_v<T>;
#endif


#ifdef __cpp_concepts
    template<typename T>
    concept eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#else
    /// Whether an object is a regular Eigen matrix.
    template<typename T>
    struct is_eigen_matrix : std::bool_constant<is_eigen_native_v<T> or is_eigen_zero_expr_v<T>> {};

    /// Helper template for is_eigen_native.
    template<typename T>
    inline constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;

    /// Synonym for is_eigen_matrix_v
    template<typename T>
    inline constexpr bool eigen_matrix = is_eigen_matrix_v<T>;
#endif

/////////////////
//    Other    //
/////////////////

    namespace internal
    {
      /*
       * Base class for all OpenKalman classes with a base that is an Eigen3 matrix.
       */
      template<typename Derived, typename Nested>
      struct Eigen3MatrixBase;

      /*
       * Base class for Covariance and SquareRootCovariance with a base that is an Eigen3 matrix.
       */
      template<typename Derived, typename Nested, typename Enable = void>
      struct Eigen3CovarianceBase;

    } // namespace internal

  } // namespace Eigen3


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires (is_triangular_v<T> or is_self_adjoint_v<T> )
  struct is_covariance_base<T> : std::true_type {};
#else
  template<typename T>
    struct is_covariance_base<T,
      std::enable_if_t<Eigen3::is_eigen_native_v<T> and (is_triangular_v<T> or is_self_adjoint_v<T>)>>
      : std::true_type {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T>
  struct is_typed_matrix_base<T> : std::true_type {};
#else
  template<typename T>
    struct is_typed_matrix_base<T, std::enable_if_t<Eigen3::is_eigen_native_v<T>>> : std::true_type {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_gettable<T, 2> : std::true_type {};
#else
  template<typename T>
    struct is_element_gettable<T, 2, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
      Eigen3::is_eigen_native_v<T>>>
      : std::true_type {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_gettable<T, 1> : std::bool_constant<MatrixTraits<T>::columns == 1> {};
#else
  template<typename T>
    struct is_element_gettable<T, 1, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
      Eigen3::is_eigen_native_v<T>>>
      : std::bool_constant<MatrixTraits<T>::columns == 1> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_settable<T, 2>
    : std::bool_constant<not std::is_const_v<std::remove_reference_t<T>> and
      static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};
#else
  template<typename T>
    struct is_element_settable<T, 2, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
      Eigen3::is_eigen_native_v<T>>>
      : std::bool_constant<not std::is_const_v<std::remove_reference_t<T>> and
        static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::is_same_v<T, std::decay_t<T>>
  struct is_element_settable<T, 1>
    : std::bool_constant<MatrixTraits<T>::columns == 1 and not std::is_const_v<std::remove_reference_t<T>> and
      static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};
#else
  template<typename T>
    struct is_element_settable<T, 1, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
      Eigen3::is_eigen_native_v<T>>>
      : std::bool_constant<MatrixTraits<T>::columns == 1 and not std::is_const_v<std::remove_reference_t<T>> and
        static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};
#endif


  /////////////////////////////
  //    SelfAdjointMatrix    //
  /////////////////////////////

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_covariance_base<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_typed_matrix_base<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_zero<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_identity<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::bool_constant<is_identity_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_diagonal<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::bool_constant<is_diagonal_v<BaseMatrix> or storage_triangle == TriangleType::diagonal> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_self_adjoint<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix> and storage_triangle != TriangleType::diagonal>>
    : std::true_type {};

  namespace Eigen3
  {
    template<typename T>
    struct is_upper_storage_triangle : OpenKalman::internal::class_trait<is_upper_storage_triangle, T> {};

    template<typename BaseMatrix>
    struct is_upper_storage_triangle<Eigen3::SelfAdjointMatrix<BaseMatrix, TriangleType::upper>>
      : std::true_type {};

    template<typename T>
    inline constexpr bool is_upper_storage_triangle_v = is_upper_storage_triangle<T>::value;

    template<typename T>
    struct is_lower_storage_triangle : OpenKalman::internal::class_trait<is_lower_storage_triangle, T> {};

    template<typename BaseMatrix>
    struct is_lower_storage_triangle<Eigen3::SelfAdjointMatrix<BaseMatrix, TriangleType::lower>>
      : std::true_type {};

    template<typename T>
    inline constexpr bool is_lower_storage_triangle_v = is_lower_storage_triangle<T>::value;
  } // namespace Eigen3


  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_strict<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_gettable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 2>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 2> or is_element_gettable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_gettable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 1>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 1> or
      (is_element_gettable_v<BaseMatrix, 2> and storage_triangle == TriangleType::diagonal)> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_settable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 2>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 2> or is_element_settable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_settable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 1>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 1> or
      (is_element_settable_v<BaseMatrix, 2> and storage_triangle == TriangleType::diagonal)> {};


  ////////////////////////////
  //    TriangularMatrix    //
  ////////////////////////////

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_covariance_base<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_typed_matrix_base<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_zero<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_identity<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>>
    : std::bool_constant<is_identity_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_diagonal<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::bool_constant<is_diagonal_v<BaseMatrix> or triangle_type == TriangleType::diagonal> {};

  template<typename BaseMatrix>
  struct is_lower_triangular<Eigen3::TriangularMatrix<BaseMatrix, TriangleType::lower>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_upper_triangular<Eigen3::TriangularMatrix<BaseMatrix, TriangleType::upper>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_strict<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_gettable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 2>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 2> or is_element_gettable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_gettable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 1>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 1> or
      (is_element_gettable_v<BaseMatrix, 2> and triangle_type == TriangleType::diagonal)> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_settable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 2>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 2> or is_element_settable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_settable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 1>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 1> or
      (is_element_settable_v<BaseMatrix, 2> and triangle_type == TriangleType::diagonal)> {};


  // -------------------- //
  //    DiagonalMatrix    //
  // -------------------- //

  template<typename BaseMatrix>
  struct is_covariance_base<Eigen3::DiagonalMatrix<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_typed_matrix_base<Eigen3::DiagonalMatrix<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_zero<Eigen3::DiagonalMatrix<BaseMatrix>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix>
  struct is_diagonal<Eigen3::DiagonalMatrix<BaseMatrix>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_strict<Eigen3::DiagonalMatrix<BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_gettable_v<BaseMatrix, 1> or is_element_gettable_v<BaseMatrix, 2>)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_settable_v<BaseMatrix, 1> or is_element_settable_v<BaseMatrix, 2>)> {};


  // ------------------------------------------------------- //
  //    ZeroMatrix and other known Eigen zero expressions    //
  // ------------------------------------------------------- //

  template<typename BaseMatrix>
  struct is_zero<Eigen3::ZeroMatrix<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_covariance_base<Eigen3::ZeroMatrix<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_typed_matrix_base<Eigen3::ZeroMatrix<BaseMatrix>> : std::true_type {};

  template<typename ArgType>
  struct is_strict<Eigen3::ZeroMatrix<ArgType>> : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> or is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> or is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> and is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<(is_zero_v<Arg1> and is_zero_v<Arg2>) or (is_identity_v<Arg1> and is_identity_v<Arg2>)>>
    : std::true_type {};

  template<typename Arg>
  struct is_zero<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<is_zero_v<Arg>>>
    : std::true_type {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ZeroMatrix<BaseMatrix>, N>
    : std::bool_constant<N == 2 or (N == 1 and MatrixTraits<BaseMatrix>::columns == 1)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ZeroMatrix<BaseMatrix>, N> : std::false_type {};


  ///////////////////////////
  //    ToEuclideanExpr    //
  ///////////////////////////

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix_base<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<Coefficients::axes_only and is_element_settable_v<BaseMatrix, N>> {};


  /////////////////////////////
  //    FromEuclideanExpr    //
  /////////////////////////////

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix_base<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<(Coefficients::axes_only and is_element_settable_v<BaseMatrix, N>) or
    (Eigen3::to_euclidean_expr<BaseMatrix> and is_element_settable_v<typename MatrixTraits<BaseMatrix>::BaseMatrix, N>)>
    {};


  ///////////////////////////////////////////////////////////////
  //    Eigen Identity and other known diagonal expressions    //
  ///////////////////////////////////////////////////////////////

  template<typename Arg>
  using EigenIdentity = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename Arg::Scalar>, Arg>;

  template<typename Arg>
  struct is_identity<EigenIdentity<Arg>>
    : std::bool_constant<Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};

  /// Product of two identity matrices is also identity.
  template<typename Arg1, typename Arg2>
  struct is_identity<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<is_identity_v<Arg1> and is_identity_v<Arg2>> {};

  /// Product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<not((is_zero_v<Arg1> or is_zero_v<Arg2>) or
      (is_identity_v<Arg1> and is_identity_v<Arg2>) or
      (Arg1::RowsAtCompileTime == 1 and Arg2::ColsAtCompileTime == 1))>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Diagonal matrix times a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_1by1_v<Arg1> and not is_1by1_v<Arg2>>>
    : std::bool_constant<is_diagonal_v<Arg1> or is_diagonal_v<Arg2>> {};

  /// Diagonal matrix divided by a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1>>>
    : std::bool_constant<is_diagonal_v<Arg1>> {};

  /// Sum of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<
      not (is_zero_v<Arg1> and is_zero_v<Arg2>) and
      not (is_1by1_v<Arg1> and is_1by1_v<Arg2>)>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Difference of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<
      not (is_zero_v<Arg1> and is_zero_v<Arg2>) and
      not (is_identity_v<Arg1> and is_identity_v<Arg2>) and
      not (is_1by1_v<Arg1> and is_1by1_v<Arg2>)>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// The negation of an identity matrix is diagonal.
  template<typename Arg>
  struct is_diagonal<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<not is_zero_v<Arg> and not is_1by1_v<Arg>>>
    : std::bool_constant<is_diagonal_v<Arg>> {};


  // ------------ //
  //    Matrix    //
  // ------------ //

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  // Default for Eigen: the base matrix will be an Eigen::Matrix of the appropriate size.
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients = RowCoefficients,
    typename BaseMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>> requires
  is_typed_matrix_base_v<BaseMatrix> and (RowCoefficients::size == MatrixTraits<BaseMatrix>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<BaseMatrix>::columns)
#else
  template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
    typename BaseMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
#endif
  struct Matrix;
#endif


#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  /// If the arguments are a sequence of scalars, deduce a single-column matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Matrix(Args ...) -> Matrix<Axes<sizeof...(Args)>, Axis,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;
#endif


  namespace Eigen3
  {
    /// Make Mean from a list of coefficients.
    template<
      typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_Matrix(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = RowCoefficients::size;
      constexpr auto cols = ColumnCoefficients::size;
      static_assert(dim * cols == sizeof...(Args));
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }

    /// Make Mean from a list of coefficients.
    template<
      typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic_v<Args>...>, int> = 0>
    auto make_Matrix(Args ... args)
    {
      using Coeffs = Axes<sizeof...(Args)>;
      return make_Matrix<Coeffs, Coefficients<Axis>>(args...);
    }

    /// Make Mean from a Scalar type and one or two sets of Coefficients.
    template<
      typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
      std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0,
      std::enable_if_t<not std::is_arithmetic_v<RowCoefficients>, int> = 0,
      std::enable_if_t<not std::is_arithmetic_v<ColumnCoefficients>, int> = 0>
    auto make_Matrix()
    {
      using Mat = Eigen::Matrix<Scalar, RowCoefficients::size, ColumnCoefficients::size>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>();
    }
  }

  // ---------- //
  //    Mean    //
  // ---------- //

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the Coefficients.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::size, 1>> requires
    is_typed_matrix_base_v<BaseMatrix> and (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
#endif
  struct Mean;
#endif


#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  /// If the arguments are a sequence of scalars, deduce a single-column mean with all Axis coefficients.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Mean(Args ...) -> Mean<Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;
#endif

  namespace Eigen3
  {
    /// Make Mean from a list of coefficients, if Coefficients types are known.
    template<
      typename Coefficients, typename ... Args,
      std::enable_if_t<not std::is_arithmetic_v<Coefficients> and (sizeof...(Args) > 0) and
        std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_Mean(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr std::size_t dim = Coefficients::size;
      static_assert(sizeof...(Args) % dim == 0);
      constexpr auto cols = sizeof...(Args) / dim;
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return Mean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }

    /// Make Mean from a list of coefficients, assuming that Coefficients types are all Axis.
    template<
      typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_Mean(Args ... args)
    {
      return make_Mean<OpenKalman::Axes<sizeof...(Args)>>(args...);
    }

    /// Make a default Eigen3 Mean, based on a Scalar type, a set of Coefficients, and a number of columns.
    template<
      typename Scalar, typename Coefficients, std::size_t cols = 1,
      std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0,
      std::enable_if_t<not std::is_arithmetic_v<Coefficients>, int> = 0>
    auto make_Mean()
    {
      return Mean<Coefficients, Eigen::Matrix<Scalar, Coefficients::size, cols>>();
    }
  }


  // ------------------- //
  //    EuclideanMean    //
  // ------------------- //

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>> requires
    is_typed_matrix_base_v<BaseMatrix> and (Coefficients::dimension == MatrixTraits<BaseMatrix>::dimension)
  struct EuclideanMean;
#else
  template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
  struct EuclideanMean;
#endif
#endif


#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  EuclideanMean(Args ...) -> EuclideanMean<OpenKalman::Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;
#endif


  namespace Eigen3
  {
    /// Make Euclidean mean from a list of coefficients.
    template<
      typename Coefficients,
      typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_EuclideanMean(Args ... args) noexcept
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::dimension;
      static_assert(sizeof...(Args) % dim == 0);
      using Mat = Eigen::Matrix<Scalar, dim, 1>;
      return EuclideanMean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }

    /// Make Mean from a list of coefficients.
    template<
      typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_EuclideanMean(Args ... args) noexcept
    {
      using Coefficients = OpenKalman::Axes<sizeof...(Args)>;
      return make_EuclideanMean<Coefficients>(args...);
    }

    /// Make strict EuclideanMean from a Scalar type, a set of Coefficients, and a number of columns.
    template<
      typename Scalar, typename Coefficients, std::size_t cols = 1,
      std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0,
      std::enable_if_t<not std::is_arithmetic_v<Coefficients>, int> = 0>
    auto make_EuclideanMean()
    {
      using Mat = Eigen::Matrix<Scalar, Coefficients::dimension, cols>;
      return Mean<Coefficients, Mat>();
    }
  }


  //////////////////////
  //    Covariance    //
  //////////////////////

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  template<
    typename Coefficients,
    typename ArgType = Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  struct Covariance;
#endif


#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Covariance(Args ...) -> Covariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
  Eigen3::SelfAdjointMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
    internal::constexpr_sqrt(sizeof...(Args)), internal::constexpr_sqrt(sizeof...(Args))>>>;
#endif


  namespace Eigen3
  {
    /// Make a Covariance, based on a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type, typename ... Args> requires
      (sizeof...(Args) > 0) and (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<
      typename Coefficients, TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and is_coefficients_v < Coefficients>and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Covariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::size;
      static_assert(sizeof...(Args) == dim * dim);
      using Mat = Eigen::Matrix<Scalar, dim, dim>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type...>;
      using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
      return Covariance<Coefficients, B>(MatrixTraits<SA>::make(args...));
    }

    /// Make an axes-only covariance, based on a list of coefficients in row-major order.
    template<
      TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and
        std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_Covariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      static_assert(sizeof...(Args) == dim * dim);
      using Coefficients = OpenKalman::Axes<dim>;
      return make_Covariance<Coefficients, triangle_type...>(args...);
    }

    /// Make default Covariance, based the size on the number of coefficients.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type> requires (sizeof...(triangle_type) <= 1)
#else
    template<typename Coefficients, TriangleType ... triangle_type, std::enable_if_t<
      sizeof...(triangle_type) <= 1 and is_coefficients_v < Coefficients>, int> = 0>
#endif
    auto make_Covariance()
    {
      using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type...>;
      using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
      return Covariance<Coefficients, B>();
    }
  }


  ////////////////////////////////
  //    SquareRootCovariance    //
  ////////////////////////////////

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
  template<
    typename Coefficients,
    typename ArgType = Eigen3::TriangularMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  struct SquareRootCovariance;
#endif


#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE
    /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  SquareRootCovariance(Args ...) -> SquareRootCovariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
  Eigen3::TriangularMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
    OpenKalman::internal::constexpr_sqrt(sizeof...(Args)), OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>>>;
#endif


  namespace Eigen3
  {
    /// Make SquareRootCovariance matrix using a list of coefficients in row-major order representing a triangular matrix.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type, typename ... Args> requires
      (sizeof...(Args) > 0) and (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename Coefficients, TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and is_coefficients_v<Coefficients> and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_SquareRootCovariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::size;
      static_assert(sizeof...(Args) == dim * dim);
      using Mat = Eigen::Matrix<Scalar, dim, dim>;
      using B = std::conditional_t<(sizeof...(triangle_type) == 1), // Is triangle type specified?
        typename MatrixTraits<Mat>::template TriangularBaseType<triangle_type...>,
        typename MatrixTraits<Mat>::template TriangularBaseType<TriangleType::lower>>; // lower-triangular self-adjoint, by default
      return SquareRootCovariance<Coefficients, B>(MatrixTraits<Mat>::make(args...));
    }

    /// Make an axes-only covariance, based on a list of coefficients in row-major order.
    template<TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and
        std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
    auto make_SquareRootCovariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      static_assert(sizeof...(Args) == dim * dim);
      using Coefficients = OpenKalman::Axes<dim>;
      return make_SquareRootCovariance<Coefficients, triangle_type...>(args...);
    }

    /// Make default Covariance, based the size on the number of coefficients.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type> requires (sizeof...(triangle_type) <= 1)
#else
    template<typename Coefficients, TriangleType ... triangle_type,
      std::enable_if_t<sizeof...(triangle_type) <= 1 and is_coefficients_v<Coefficients>, int> = 0>
#endif
    auto make_SquareRootCovariance()
    {
      using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
      using B = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      return SquareRootCovariance<Coefficients, B>();
    }
  } // namespace Eigen3

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
