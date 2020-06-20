/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENFORWARDDECLARATIONS_H
#define OPENKALMAN_EIGENFORWARDDECLARATIONS_H

#include <type_traits>

namespace OpenKalman
{

  namespace internal
  {
    namespace detail
    {
      template<typename T>
      constexpr T sqrt_impl(T x, T lo, T hi)
      {
        if (lo == hi) return lo;
        const T mid = (lo + hi + 1) / 2;
        if (x / mid < mid) return sqrt_impl<T>(x, lo, mid - 1);
        else return sqrt_impl(x, mid, hi);
      }
    }

    template<typename T>
    constexpr T constexpr_sqrt(T x)
    {
      return detail::sqrt_impl<T>(x, 0, x / 2 + 1);
    }
  }

  ///////////////////////////
  //    ToEuclideanExpr    //
  ///////////////////////////

  template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
  struct ToEuclideanExpr;

  template<typename T>
  struct is_ToEuclideanExpr : class_trait<is_ToEuclideanExpr, T> {};

  template<typename T>
  inline constexpr bool is_ToEuclideanExpr_v = is_ToEuclideanExpr<T>::value;

  template<typename Coefficients, typename BaseMatrix>
  struct is_ToEuclideanExpr<ToEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<ToEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix_base<ToEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};


  /////////////////////////////
  //    FromEuclideanExpr    //
  /////////////////////////////

  template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
  struct FromEuclideanExpr;

  template<typename T>
  struct is_FromEuclideanExpr : class_trait<is_FromEuclideanExpr, T> {};

  template<typename T>
  inline constexpr bool is_FromEuclideanExpr_v = is_FromEuclideanExpr<T>::value;

  template<typename Coefficients, typename BaseMatrix>
  struct is_FromEuclideanExpr<FromEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<FromEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix_base<FromEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};


  //////////////////////////////////
  //    EigenSelfAdjointMatrix    //
  //////////////////////////////////

  template<typename BaseMatrix, TriangleType storage_triangle = TriangleType::lower>
  struct EigenSelfAdjointMatrix;

  template<typename T>
  struct is_EigenSelfAdjointMatrix : class_trait<is_EigenSelfAdjointMatrix, T> {};

  template<typename T>
  inline constexpr bool is_EigenSelfAdjointMatrix_v = is_EigenSelfAdjointMatrix<T>::value;

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_EigenSelfAdjointMatrix<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>> : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_covariance_base<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>> : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_typed_matrix_base<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>> : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_zero<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>, std::enable_if_t<is_zero_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_diagonal<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<is_diagonal_v<BaseMatrix> and not is_zero_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_diagonal<EigenSelfAdjointMatrix<BaseMatrix, TriangleType::diagonal>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_self_adjoint<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<not OpenKalman::is_diagonal_v<BaseMatrix> and
    storage_triangle != TriangleType::diagonal>> : std::true_type {};

  template<typename T>
  struct is_Eigen_upper_storage_triangle : class_trait<is_Eigen_upper_storage_triangle, T> {};

  template<typename BaseMatrix>
  struct is_Eigen_upper_storage_triangle<EigenSelfAdjointMatrix<BaseMatrix, TriangleType::upper>> : std::true_type {};

  template<typename T>
  inline constexpr bool is_Eigen_upper_storage_triangle_v = is_Eigen_upper_storage_triangle<T>::value;

  template<typename T>
  struct is_Eigen_lower_storage_triangle : class_trait<is_Eigen_lower_storage_triangle, T> {};

  template<typename BaseMatrix>
  struct is_Eigen_lower_storage_triangle<EigenSelfAdjointMatrix<BaseMatrix, TriangleType::lower>> : std::true_type {};

  template<typename T>
  inline constexpr bool is_Eigen_lower_storage_triangle_v = is_Eigen_lower_storage_triangle<T>::value;

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_strict<EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>> : is_strict<BaseMatrix> {};


  /////////////////////////////////
  //    EigenTriangularMatrix    //
  /////////////////////////////////

  template<typename BaseMatrix, TriangleType triangle_type = TriangleType::lower>
  struct EigenTriangularMatrix;

  template<typename T>
  struct is_EigenTriangularMatrix : class_trait<is_EigenTriangularMatrix, T> {};

  template<typename T>
  inline constexpr bool is_EigenTriangularMatrix_v = is_EigenTriangularMatrix<T>::value;

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_EigenTriangularMatrix<EigenTriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_covariance_base<EigenTriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_typed_matrix_base<EigenTriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_zero<EigenTriangularMatrix<BaseMatrix, triangle_type>, std::enable_if_t<is_zero_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_diagonal<EigenTriangularMatrix<BaseMatrix, triangle_type>,
    std::enable_if_t<is_diagonal_v<BaseMatrix> and not is_zero_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_diagonal<EigenTriangularMatrix<BaseMatrix, TriangleType::diagonal>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
  : std::true_type {};

  template<typename BaseMatrix>
  struct is_lower_triangular<EigenTriangularMatrix<BaseMatrix, TriangleType::lower>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_upper_triangular<EigenTriangularMatrix<BaseMatrix, TriangleType::upper>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>> : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_strict<EigenTriangularMatrix<BaseMatrix, triangle_type>> : is_strict<BaseMatrix> {};


  /////////////////////////
  //    EigenDiagonal    //
  /////////////////////////

  /// A diagonal matrix. Works similarly to Eigen::DiagonalMatrix.
  template<typename BaseMatrix>
  struct EigenDiagonal;

  template<typename T>
  struct is_EigenDiagonal : class_trait<is_EigenDiagonal, T> {};

  template<typename T>
  inline constexpr bool is_EigenDiagonal_v = is_EigenDiagonal<T>::value;

  template<typename BaseMatrix>
  struct is_EigenDiagonal<EigenDiagonal<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_covariance_base<EigenDiagonal<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_typed_matrix_base<EigenDiagonal<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_diagonal<EigenDiagonal<BaseMatrix>,
    std::enable_if_t<not OpenKalman::is_zero_v<EigenDiagonal<BaseMatrix>>>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_zero<EigenDiagonal<BaseMatrix>, std::enable_if_t<OpenKalman::is_zero_v<BaseMatrix>>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_strict<EigenDiagonal<BaseMatrix>> : is_strict<BaseMatrix> {};


  ////////////////////////////////////////////////////////////
  //    EigenZero and other known Eigen zero expressions    //
  ////////////////////////////////////////////////////////////

  /// Exclusive wrapper type for zero. Necessary because Eigen3 does not distinguish between zero type and a constant type.
  /// This will be treated as a native Eigen matrix.
  template<typename ArgType>
  struct EigenZero;

  template<typename T>
  struct is_EigenZero : class_trait<is_EigenZero, T> {};

  template<typename T>
  inline constexpr bool is_EigenZero_v = is_EigenZero<T>::value;

  template<typename BaseMatrix>
  struct is_EigenZero<EigenZero<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_zero<EigenZero<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_covariance_base<EigenZero<BaseMatrix>> : std::true_type {};

  template<typename BaseMatrix>
  struct is_typed_matrix_base<EigenZero<BaseMatrix>> : std::true_type {};

  template<typename ArgType>
  struct is_strict<EigenZero<ArgType>> : std::true_type {};

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


  ///////////////////////////////////////////////////////////////
  //    Eigen Identity and other known diagonal expressions    //
  ///////////////////////////////////////////////////////////////

  template<typename Arg>
  using EigenIdentity = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename Arg::Scalar>, Arg>;

  template<typename Arg>
  struct is_identity<EigenIdentity<Arg>>
    : std::integral_constant<bool, Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};

  /// Product of two identity matrices is also identity.
  template<typename Arg1, typename Arg2>
  struct is_identity<Eigen::Product<Arg1, Arg2>>
    : std::integral_constant<bool, is_identity_v<Arg1> and is_identity_v<Arg2>> {};

  /// Product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<(not is_zero_v<Arg1> or not is_zero_v<Arg2>) and (not is_identity_v<Arg1> or not is_identity_v<Arg2>)>>
    : std::integral_constant<bool, is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Diagonal matrix times a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1> and not is_zero_v<Arg2>>>
    : std::integral_constant<bool, is_diagonal_v<Arg1> or is_diagonal_v<Arg2>> {};

  /// Diagonal matrix divided by a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1>>>
    : std::integral_constant<bool, is_diagonal_v<Arg1>> {};

  /// Sum of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1> or not is_zero_v<Arg2>>>
    : std::integral_constant<bool, is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Difference of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<(not is_zero_v<Arg1> or not is_zero_v<Arg2>) and
      (not is_identity_v<Arg1> or not is_identity_v<Arg2>) and
      (not is_1by1_v<Arg1> and not is_1by1_v<Arg2>)>>
    : std::integral_constant<bool, is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// The negation of an identity matrix is diagonal.
  template<typename Arg>
  struct is_diagonal<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<not is_zero_v<Arg> and not is_1by1_v<Arg>>>
    : std::integral_constant<bool, is_diagonal_v<Arg>> {};


  //////////////////////////
  //    General traits    //
  //////////////////////////

  /// Whether an object is a native Eigen::MatrixBase type in Eigen3.
  template<typename T>
  struct is_native_Eigen_type : std::integral_constant<bool,
    std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
    not OpenKalman::is_EigenSelfAdjointMatrix_v<T> and
    not OpenKalman::is_EigenTriangularMatrix_v<T> and
    not OpenKalman::is_EigenDiagonal_v<T> and
    not OpenKalman::is_EigenZero_v<T> and
    not OpenKalman::is_FromEuclideanExpr_v<T> and
    not OpenKalman::is_ToEuclideanExpr_v<T> and
    not OpenKalman::is_typed_matrix_v<T> and
    not OpenKalman::is_covariance_v<T>> {};

  /// Helper template for is_native_Eigen_type.
  template<typename T>
  inline constexpr bool is_native_Eigen_type_v = is_native_Eigen_type<T>::value;

  /// Whether an object is a regular Eigen matrix.
  template<typename T>
  struct is_Eigen_matrix : std::integral_constant<bool,
    OpenKalman::is_native_Eigen_type_v<T> or
    OpenKalman::is_EigenZero_v<T>> {};

  /// Helper template for is_native_Eigen_type.
  template<typename T>
  inline constexpr bool is_Eigen_matrix_v = is_Eigen_matrix<T>::value;

  template<typename T>
  struct is_covariance_base<T,
    std::enable_if_t<is_native_Eigen_type_v<T> and (is_triangular_v<T> or is_self_adjoint_v<T>)>>
    : std::true_type {};

  template<typename T>
  struct is_typed_matrix_base<T,
    std::enable_if_t<is_native_Eigen_type_v<T>>>
    : std::true_type {};


  ////////////////
  //    Mean    //
  ////////////////

  /// By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the Coefficients.
  template<typename Coefficients, typename BaseMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
  struct Mean;

  /// If the arguments are a sequence of scalars, deduce a single-column Euclidean mean.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  Mean(Args ...) -> Mean<OpenKalman::Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;

  /// Make Mean from a list of coefficients, if Coefficients types are known.
  template<
    typename Coefficients, typename ... Args,
    std::enable_if_t<not std::is_arithmetic_v<Coefficients>, int> = 0,
    std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_Mean(Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    constexpr std::size_t dim = Coefficients::size;
    static_assert(sizeof...(Args) % dim == 0);
    constexpr auto cols = sizeof...(Args) / dim;
    using Mat = Eigen::Matrix<double, dim, cols>;
    return Mean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
  }

  /// Make Mean from a list of coefficients, assuming that Coefficients types are all Axis.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
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

  //////////////////
  //    Matrix    //
  //////////////////

  template<
    typename RowCoefficients,
    typename ColumnCoefficients = RowCoefficients,
    typename ArgType = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
  struct TypedMatrix;

  /// Make Mean from a list of coefficients.
  template<
    typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
    std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_Matrix(Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    constexpr auto dim = RowCoefficients::size;
    constexpr auto cols = ColumnCoefficients::size;
    static_assert(dim * cols == sizeof...(Args));
    using Mat = Eigen::Matrix<double, dim, cols>;
    return TypedMatrix<RowCoefficients, ColumnCoefficients, Mat>(MatrixTraits<Mat>::make(args...));
  }

  /// Make Mean from a list of coefficients.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_Matrix(Args ... args)
  {
    using Coefficients = OpenKalman::Axes<sizeof...(Args)>;
    return make_Mean<Coefficients, OpenKalman::Coefficients<Axis>>(args...);
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
    return TypedMatrix<RowCoefficients, ColumnCoefficients, Mat>();
  }


  /////////////////////////
  //    EuclideanMean    //
  /////////////////////////

  template<
    typename Coefficients,
    typename BaseMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
  struct EuclideanMean;

  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  EuclideanMean(Args ...) -> EuclideanMean<OpenKalman::Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;

  /// Make Euclidean mean from a list of coefficients.
  template<
    typename Coefficients,
    typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_EuclideanMean(Args ... args) noexcept
  {
    using Scalar = std::common_type_t<Args...>;
    constexpr auto dim = Coefficients::dimension;
    static_assert(sizeof...(Args) % dim == 0);
    constexpr auto cols = sizeof...(Args) / dim;
    using Mat = Eigen::Matrix<double, dim, 1>;
    return EuclideanMean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
  }

  /// Make Mean from a list of coefficients.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
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


  //////////////////////
  //    Covariance    //
  //////////////////////

  template<
    typename Coefficients,
    typename ArgType = EigenSelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  struct Covariance;

  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  Covariance(Args ...) -> Covariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
    EigenSelfAdjointMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
      internal::constexpr_sqrt(sizeof...(Args)), internal::constexpr_sqrt(sizeof...(Args))>>>;

  /// Make a Covariance, based on a list of coefficients in row-major order.
  template<typename Coefficients, TriangleType ... triangle_type, typename ... Args,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_coefficient_v<Coefficients> and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_Covariance(Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    constexpr auto dim = Coefficients::size;
    static_assert(sizeof...(Args) == dim * dim);
    using Mat = Eigen::Matrix<double, dim, dim>;
    using T = EigenTriangularMatrix<Mat, triangle_type...>;
    using SA = EigenSelfAdjointMatrix<Mat, triangle_type...>;
    using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
    return Covariance<Coefficients, B>(MatrixTraits<SA>::make(args...));
  }

  /// Make an axes-only covariance, based on a list of coefficients in row-major order.
  template<TriangleType ... triangle_type, typename ... Args,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_Covariance(Args ... args) noexcept
  {
    constexpr auto dim = internal::constexpr_sqrt(sizeof...(Args));
    static_assert(sizeof...(Args) == dim * dim);
    using Coefficients = OpenKalman::Axes<dim>;
    return make_Covariance<Coefficients, triangle_type...>(args...);
  }

  /// Make default Covariance, based the size on the number of coefficients.
  template<typename Coefficients, TriangleType ... triangle_type,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_coefficient_v<Coefficients>, int> = 0>
  auto make_Covariance()
  {
    using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
    using T = EigenTriangularMatrix<Mat, triangle_type...>;
    using SA = EigenSelfAdjointMatrix<Mat, triangle_type...>;
    using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
    return Covariance<Coefficients, B>();
  }


  ////////////////////////////////
  //    SquareRootCovariance    //
  ////////////////////////////////

  template<
    typename Coefficients,
    typename ArgType = EigenTriangularMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  struct SquareRootCovariance;

  /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  SquareRootCovariance(Args ...) -> SquareRootCovariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
    EigenTriangularMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
      internal::constexpr_sqrt(sizeof...(Args)), internal::constexpr_sqrt(sizeof...(Args))>>>;

  /// Make SquareRootCovariance matrix using a list of coefficients in row-major order representing a triangular matrix.
  /// Only the coefficients in the lower-left corner are significant.
  template<typename Coefficients, TriangleType ... triangle_type, typename ... Args,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_coefficient_v<Coefficients> and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_SquareRootCovariance(Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    constexpr auto dim = Coefficients::size;
    static_assert(sizeof...(Args) == dim * dim);
    using Mat = Eigen::Matrix<double, dim, dim>;
    using B = std::conditional_t<(sizeof...(triangle_type) == 1), // Is triangle type specified?
      typename MatrixTraits<Mat>::template TriangularBaseType<triangle_type...>,
      typename MatrixTraits<Mat>::template TriangularBaseType<TriangleType::lower>>; // lower-triangular self-adjoint, by default
    return SquareRootCovariance<Coefficients, B>(MatrixTraits<Mat>::make(args...));
  }

  /// Make an axes-only covariance, based on a list of coefficients in row-major order.
  template<TriangleType ... triangle_type, typename ... Args,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
  auto make_SquareRootCovariance(Args ... args) noexcept
  {
    constexpr auto dim = internal::constexpr_sqrt(sizeof...(Args));
    static_assert(sizeof...(Args) == dim * dim);
    using Coefficients = OpenKalman::Axes<dim>;
    return make_SquareRootCovariance<Coefficients, triangle_type...>(args...);
  }

  /// Make default Covariance, based the size on the number of coefficients.
  template<typename Coefficients, TriangleType ... triangle_type,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_coefficient_v<Coefficients>, int> = 0>
  auto make_SquareRootCovariance()
  {
    using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
    using B = EigenTriangularMatrix<Mat, triangle_type...>;
    return SquareRootCovariance<Coefficients, B>();
  }


  /////////////////
  //    Other    //
  /////////////////

  namespace internal
  {
    /*
     * Base class for all OpenKalman classes that are also Eigen3 matrices.
     */
    template<typename Derived, typename Nested>
    struct EigenMatrixBase;

    /*
     * Base class for all OpenKalman covariance classes that are also Eigen3 matrices.
     */
    template<typename Derived, typename Nested, typename Enable = void>
    struct EigenCovarianceBase;

    template<
      typename Derived,
      typename Coeffs, /// Coefficients.
      typename NestedType> /// A nested, non-Euclidean matrix.
    struct EuclideanExprBase;
  }


}

#endif //OPENKALMAN_EIGENFORWARDDECLARATIONS_H
