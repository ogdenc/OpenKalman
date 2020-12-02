/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_DEFAULT_OVERLOADS_HPP
#define OPENKALMAN_DEFAULT_OVERLOADS_HPP

/**
 * \file
 * \brief A header file defining Eigen3 as the default interface.
 * \details The definitions in this file are only enabled if Eigen3 is the first-defined interface.
 */

#include <type_traits>


namespace OpenKalman
{

  namespace Eigen3
  {
    // ---------------------------------------- //
    //  Make functions for OpenKalman matrices  //
    // ---------------------------------------- //

    /**
     * \overload OpenKalman::make_matrix
     * \brief For Eigen3: Make a Matrix from a list of coefficients, specifying the row and column coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam RowCoefficients The coefficient types corresponding to the rows.
     * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal RowCoefficients::size * ColumnCoefficients::size.
     */
#ifdef __cpp_concepts
    template<coefficients RowCoefficients, coefficients ColumnCoefficients = RowCoefficients, typename ... Args>
    requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
    (RowCoefficients::size * ColumnCoefficients::size == sizeof...(Args))
#else
    template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...> and
      (RowCoefficients::size * ColumnCoefficients::size == sizeof...(Args)), int> = 0>
#endif
    auto make_matrix(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      using Mat = Eigen::Matrix<Scalar, RowCoefficients::size, ColumnCoefficients::size>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column Matrix from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_matrix(Args ... args)
    {
      return make_matrix<Axes<sizeof...(Args)>, Axis>(args...);
    }


    /**
     * \overload
     * \brief For Eigen3: Make a Matrix based on a scalar type and row and column coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Scalar A scalar type (integral or floating-point).
     * \tparam RowCoefficients The coefficient types corresponding to the rows.
     * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
     */
#ifdef __cpp_concepts
    template<typename Scalar, coefficients RowCoefficients, coefficients ColumnCoefficients = RowCoefficients> requires
    std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
    std::enable_if_t<std::is_arithmetic_v<Scalar> and
      coefficients<RowCoefficients> and coefficients<ColumnCoefficients>, int> = 0>
#endif
    auto make_matrix()
    {
      using Mat = Eigen::Matrix<Scalar, RowCoefficients::size, ColumnCoefficients::size>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>();
    }


    /**
     * \overload OpenKalman::make_mean
     * \brief For Eigen3: Make a Mean from a list of coefficients, specifying the row coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be divisible by Coefficients::size.
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename ... Args> requires
      (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and (sizeof...(Args) % Coefficients::size == 0)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<coefficients<Coefficients> and
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...> and
      (sizeof...(Args) % Coefficients::size == 0), int> = 0>
#endif
    auto make_mean(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr std::size_t dim = Coefficients::size;
      constexpr std::size_t cols = sizeof...(Args) / dim;
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return Mean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column Mean from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_mean(Args ... args)
    {
      return make_mean<OpenKalman::Axes<sizeof...(Args)>>(args...);
    }


    /**
     * \overload
     * \brief For Eigen3: Make a Mean based on a scalar type, a set of row coefficients, and a number of columns.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Scalar A scalar type (integral or floating-point).
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam cols The number of columns.
     */
#ifdef __cpp_concepts
    template<typename Scalar, coefficients Coefficients, std::size_t cols = 1> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
    std::is_arithmetic_v<Scalar> and coefficients<Coefficients>, int> = 0>
#endif
    auto make_mean()
    {
      return Mean<Coefficients, Eigen::Matrix<Scalar, Coefficients::size, cols>>();
    }


    /**
     * \overload OpenKalman::make_euclidean_mean
     * \brief For Eigen3: Make a EuclideanMean from a list of coefficients, specifying the row coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be divisible by Coefficients::dimension.
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename ... Args> requires
    (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and (sizeof...(Args) % Coefficients::dimension == 0)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...> and
      (sizeof...(Args) % Coefficients::dimension == 0), int> = 0>
#endif
    auto make_euclidean_mean(Args ... args) noexcept
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr std::size_t dim = Coefficients::dimension;
      constexpr std::size_t cols = sizeof...(Args) / dim;
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return EuclideanMean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column EuclideanMean from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_euclidean_mean(Args ... args) noexcept
    {
      return make_euclidean_mean<Axes<sizeof...(Args)>>(args...);
    }


    /**
     * \overload
     * \brief For Eigen3: Make a EuclideanMean based on a scalar type, row coefficients, and a number of columns.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Scalar A scalar type (integral or floating-point).
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam cols The number of columns.
     */
#ifdef __cpp_concepts
    template<typename Scalar, coefficients Coefficients, std::size_t cols = 1> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
    std::is_arithmetic_v<Scalar> and coefficients<Coefficients>, int> = 0>
#endif
    auto make_euclidean_mean()
    {
      return EuclideanMean<Coefficients,  Eigen::Matrix<Scalar, Coefficients::dimension, cols>>();
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a Covariance, with nested triangular type, from a list of coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal Coefficients::size * Coefficients::size.
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType triangle_type, typename ... Args> requires
    (std::is_arithmetic_v<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size)
#else
    template<typename Coefficients, TriangleType triangle_type, typename ... Args, std::enable_if_t<
      coefficients<Coefficients> and std::conjunction_v<std::is_arithmetic<Args>...> and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size), int> = 0>
#endif
    auto make_covariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      using Mat = Eigen::Matrix<Scalar, Coefficients::size, Coefficients::size>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type>;
      return Covariance<Coefficients, T>(MatrixTraits<SA>::make(args...));
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a Covariance from a list of coefficients, specifying the coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal Coefficients::size * Coefficients::size.
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename ... Args> requires
    (std::is_arithmetic_v<Args> and ...) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<
      coefficients<Coefficients> and std::conjunction_v<std::is_arithmetic<Args>...> and
      (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size), int> = 0>
#endif
    auto make_covariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      using Mat = Eigen::Matrix<Scalar, Coefficients::size, Coefficients::size>;
      using SA = Eigen3::SelfAdjointMatrix<Mat>;
      return Covariance<Coefficients, SA>(MatrixTraits<SA>::make(args...));
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a default Axis Covariance, with nested triangular type, from a list of coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be the square of an integer.
     */
#ifdef __cpp_concepts
    template<TriangleType triangle_type, typename ... Args> requires
      (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<TriangleType triangle_type, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_covariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Axes<dim>;
      return make_covariance<Coefficients, triangle_type>(args...);
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a Covariance from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be the square of an integer.
     */
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
    (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
      OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_covariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Axes<dim>;
      return make_covariance<Coefficients>(args...);
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a writable, uninitialized Covariance with nested triangular matrix.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Scalar The scalar type (integral or floating-point).
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType triangle_type, typename Scalar = double> requires
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      std::is_arithmetic_v<Scalar>
#else
    template<typename Coefficients, TriangleType triangle_type, typename Scalar = double, std::enable_if_t<
      coefficients<Coefficients> and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      std::is_arithmetic_v<Scalar>, int> = 0>
#endif
    auto make_covariance()
    {
      using Mat = Eigen::Matrix<Scalar, Coefficients::size, Coefficients::size>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type>;
      return Covariance<Coefficients, T>();
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a writable, uninitialized Covariance, specifying the coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam Scalar The scalar type (integral or floating-point).
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename Scalar = double> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Coefficients, typename Scalar = double, std::enable_if_t<
      coefficients<Coefficients> and std::is_arithmetic_v<Scalar>, int> = 0>
#endif
    auto make_covariance()
    {
      using Mat = Eigen::Matrix<Scalar, Coefficients::size, Coefficients::size>;
      using SA = Eigen3::SelfAdjointMatrix<Mat>;
      return Covariance<Coefficients, SA>();
    }


    /**
     * \overload OpenKalman::make_square_root_covariance
     * \brief For Eigen3: Make a SquareRootCovariance from a list of coefficients.
     * \details Only the coefficients in the associated upper or lower triangle are significant.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal Coefficients::size * Coefficients::size.
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType triangle_type = TriangleType::lower, typename ... Args> requires
      (std::is_arithmetic_v<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size)
#else
    template<typename Coefficients, TriangleType triangle_type = TriangleType::lower, typename ... Args,
      std::enable_if_t<coefficients<Coefficients> and std::conjunction_v<std::is_arithmetic<Args>...> and
        (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
        (sizeof...(Args) > 0) and (sizeof...(Args) == Coefficients::size * Coefficients::size), int> = 0>
#endif
    auto make_square_root_covariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      using Mat = Eigen::Matrix<Scalar, Coefficients::size, Coefficients::size>;
      using T = typename MatrixTraits<Mat>::template TriangularBaseType<triangle_type>;
      return SquareRootCovariance<Coefficients, T>(MatrixTraits<T>::make(args...));
    }


    /**
     * \overload OpenKalman::make_square_root_covariance
     * \brief For Eigen3: Make a default Axis SquareRootCovariance from a list of coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be the square of an integer.
     */
#ifdef __cpp_concepts
    template<TriangleType triangle_type = TriangleType::lower, typename ... Args> requires
    (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<TriangleType triangle_type = TriangleType::lower, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_square_root_covariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Axes<dim>;
      return make_square_root_covariance<Coefficients, triangle_type>(args...);
    }


    /**
     * \overload OpenKalman::make_square_root_covariance
     * \brief For Eigen3: Make a writable, uninitialized SquareRootCovariance.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Scalar The scalar type (integral or floating-point).
     */
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType triangle_type = TriangleType::lower, typename Scalar = double>
    requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    std::is_arithmetic_v<Scalar>
#else
    template<typename Coefficients, TriangleType triangle_type = TriangleType::lower, typename Scalar = double,
      std::enable_if_t<coefficients<Coefficients> and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      std::is_arithmetic_v<Scalar>, int> = 0>
#endif
    auto make_square_root_covariance()
    {
      using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type>;
      return SquareRootCovariance<Coefficients, T>();
    }

  } // namespace Eigen3


  // --------------------------------------------------------------------------- //
  //  The following are only included if Eigen3 is the first-included interface  //
  // --------------------------------------------------------------------------- //

#ifndef OPENKALMAN_FIRST_INTERFACE
/**
 * \internal
 * \brief Specifies the first-defined interface to a matrix library (e.g., Eigen3).
 * \details The first-defined interface will define defaults such as the default nested matrices for
 * library classes like Matrix, and the deduction guides for constructors that take only scalar coefficients.
 */
#define OPENKALMAN_FIRST_INTERFACE

  // Import make functions into the main namespace
  using Eigen3::make_matrix;
  using Eigen3::make_mean;
  using Eigen3::make_euclidean_mean;
  using Eigen3::make_covariance;
  using Eigen3::make_square_root_covariance;


  // ------------ //
  //    Matrix    //
  // ------------ //

  // Default specialization for Eigen: the nested matrix will be an Eigen::Matrix of the appropriate size.
#ifdef __cpp_concepts
  template<
    coefficients RowCoefficients,
    coefficients ColumnCoefficients = RowCoefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
  requires
    (RowCoefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<NestedMatrix>::columns) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<
    typename RowCoefficients,
    typename ColumnCoefficients = RowCoefficients,
    typename NestedMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
#endif
  struct Matrix;


  // If the arguments are a sequence of scalars, deduce a single-column matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Matrix(Args ...) -> Matrix<Axes<sizeof...(Args)>, Axis,
  Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ---------- //
  //    Mean    //
  // ---------- //

  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the Coefficients.
#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
  requires (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
#endif
  struct Mean;


  /// If the arguments are a sequence of scalars, deduce a single-column mean with all Axis coefficients.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Mean(Args ...) -> Mean<Axes<sizeof...(Args)>,
  Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ------------------- //
  //    EuclideanMean    //
  // ------------------- //

#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
  requires
    (Coefficients::dimension == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
#endif
  struct EuclideanMean;


  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  EuclideanMean(Args ...) -> EuclideanMean<OpenKalman::Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ---------------- //
  //    Covariance    //
  // ---------------- //

#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  requires (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
#endif
  struct Covariance;


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


  // --------------------- //
  //  SquareRootCovariance //
  // --------------------- //

#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix =
  Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>> requires
  (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
#endif
  struct SquareRootCovariance;


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


  // -------------- //
  //  MatrixTraits  //
  // -------------- //

  /*
   * \internal
   * \brief Type traits for deriving Eigen matrices based on an arithmetic type.
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T> and (not std::is_const_v<std::remove_reference_t<T>>)
  struct MatrixTraits<T>
#else
    template<typename T>
  struct MatrixTraits<T, std::enable_if_t<std::is_arithmetic_v<T> and not std::is_const_v<std::remove_reference_t<T>>>>
#endif
  {
    using Scalar = T;

    template<std::size_t rows, std::size_t cols = 1, typename S = Scalar>
    using NativeMatrix = Eigen::Matrix<S, (Eigen::Index) rows, (Eigen::Index) cols>;

    template<TriangleType storage_triangle, std::size_t dim, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<NativeMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type, std::size_t dim, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<NativeMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<NativeMatrix<dim, 1, S>>;

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<typename Arg, std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static decltype(auto)
    make(Arg&& arg) noexcept
    {
      return std::forward<Arg>(arg);
    }

    // Make matrix from a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> Arg, std::convertible_to<const Scalar> ... Args>
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto
    make(const Arg arg, const Args ... args)
    {
      return ((NativeMatrix<sizeof...(Args)>() << arg), ... , args).finished();
    }

  };


} // namespace OpenKalman

#endif // OPENKALMAN_FIRST_INTERFACE

#endif //OPENKALMAN_DEFAULT_OVERLOADS_HPP
