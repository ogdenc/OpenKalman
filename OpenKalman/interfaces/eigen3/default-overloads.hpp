/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
  using namespace OpenKalman::internal;

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
     * must equal dimension_size_of_v<RowCoefficients> * dimension_size_of_v<ColumnCoefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor RowCoefficients, typed_index_descriptor ColumnCoefficients = RowCoefficients, scalar_type ... Args>
    requires (sizeof...(Args) > 0) and (dimension_size_of_v<RowCoefficients> * dimension_size_of_v<ColumnCoefficients> == sizeof...(Args))
#else
    template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
      (dimension_size_of_v<RowCoefficients> * dimension_size_of_v<ColumnCoefficients> == sizeof...(Args)), int> = 0>
#endif
    auto make_matrix(const Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>(MatrixTraits<Mat>::make(
        static_cast<const Scalar>(args)...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column Matrix from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
    template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
    auto make_matrix(const Args ... args)
    {
      return make_matrix<Dimensions<sizeof...(Args)>, Axis>(args...);
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
    template<scalar_type Scalar, typed_index_descriptor RowCoefficients,
      typed_index_descriptor ColumnCoefficients = RowCoefficients>
#else
    template<typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
    std::enable_if_t<scalar_type<Scalar> and
      typed_index_descriptor<RowCoefficients> and typed_index_descriptor<ColumnCoefficients>, int> = 0>
#endif
    auto make_matrix()
    {
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>();
    }


    /**
     * \overload OpenKalman::make_mean
     * \brief For Eigen3: Make a Mean from a list of coefficients, specifying the row coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be divisible by dimension_size_of_v<Coefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, scalar_type ... Args> requires
      (sizeof...(Args) > 0) and (sizeof...(Args) % dimension_size_of_v<Coefficients> == 0)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<typed_index_descriptor<Coefficients> and
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
      (sizeof...(Args) % dimension_size_of_v<Coefficients> == 0), int> = 0>
#endif
    auto make_mean(const Args ... args)
    {
      using Scalar = std::decay_t<std::common_type_t<Args...>>;
      constexpr std::size_t dim = dimension_size_of_v<Coefficients>;
      constexpr std::size_t cols = sizeof...(Args) / dim;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
      return Mean<Coefficients, Mat>(MatrixTraits<Mat>::make(static_cast<const Scalar>(args)...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column Mean from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
    auto make_mean(const Args ... args)
    {
      return make_mean<OpenKalman::Dimensions<sizeof...(Args)>>(args...);
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
    template<scalar_type Scalar, typed_index_descriptor Coefficients, std::size_t cols = 1>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
    scalar_type<Scalar> and typed_index_descriptor<Coefficients>, int> = 0>
#endif
    auto make_mean()
    {
      return Mean<Coefficients, Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, cols>>();
    }


    /**
     * \overload OpenKalman::make_euclidean_mean
     * \brief For Eigen3: Make a EuclideanMean from a list of coefficients, specifying the row coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must be divisible by euclidean_dimension_size_of_v<Coefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, scalar_type ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(Args) % euclidean_dimension_size_of_v<Coefficients> == 0)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
      (sizeof...(Args) % euclidean_dimension_size_of_v<Coefficients> == 0), int> = 0>
#endif
    auto make_euclidean_mean(const Args ... args) noexcept
    {
      using Scalar = std::decay_t<std::common_type_t<Args...>>;
      constexpr std::size_t dim = euclidean_dimension_size_of_v<Coefficients>;
      constexpr std::size_t cols = sizeof...(Args) / dim;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
      return EuclideanMean<Coefficients, Mat>(MatrixTraits<Mat>::make(static_cast<const Scalar>(args)...));
    }


    /**
     * \overload
     * \brief For Eigen3: Make a one-column EuclideanMean from a list of coefficients, with default Axis coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Args A list of numerical coefficients (either integral or floating-point).
     */
#ifdef __cpp_concepts
    template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
    auto make_euclidean_mean(const Args ... args) noexcept
    {
      return make_euclidean_mean<Dimensions<sizeof...(Args)>>(args...);
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
    template<scalar_type Scalar, typed_index_descriptor Coefficients, std::size_t cols = 1>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
    scalar_type<Scalar> and typed_index_descriptor<Coefficients>, int> = 0>
#endif
    auto make_euclidean_mean()
    {
      return EuclideanMean<Coefficients,  Eigen3::eigen_matrix_t<Scalar, euclidean_dimension_size_of_v<Coefficients>, cols>>();
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a Covariance, with nested triangular type, from a list of coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, TriangleType triangle_type, scalar_type ... Args> requires
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>)
#else
    template<typename Coefficients, TriangleType triangle_type, typename ... Args, std::enable_if_t<
      typed_index_descriptor<Coefficients> and (scalar_type<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>), int> = 0>
#endif
    auto make_covariance(const Args ... args)
    {
      using Scalar = std::decay_t<std::common_type_t<Args...>>;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type>;
      return Covariance<Coefficients, T>(MatrixTraits<SA>::make(static_cast<const Scalar>(args)...));
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a Covariance from a list of coefficients, specifying the coefficients.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
     * must equal dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, scalar_type ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<
      typed_index_descriptor<Coefficients> and (scalar_type<Args> and ...) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>), int> = 0>
#endif
    auto make_covariance(const Args ... args)
    {
      using Scalar = std::decay_t<std::common_type_t<Args...>>;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
      using SA = Eigen3::SelfAdjointMatrix<Mat>;
      return Covariance<Coefficients, SA>(MatrixTraits<SA>::make(static_cast<const Scalar>(args)...));
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
    template<TriangleType triangle_type, scalar_type ... Args> requires
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<TriangleType triangle_type, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_covariance(const Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Dimensions<dim>;
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
    template<scalar_type ... Args> requires (sizeof...(Args) > 0) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
      OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_covariance(const Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Dimensions<dim>;
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
    template<typed_index_descriptor Coefficients, TriangleType triangle_type, scalar_type Scalar = double>
    requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
    template<typename Coefficients, TriangleType triangle_type, typename Scalar = double, std::enable_if_t<
      typed_index_descriptor<Coefficients> and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      scalar_type<Scalar>, int> = 0>
#endif
    auto make_covariance()
    {
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type>;
      return Covariance<Coefficients, T>();
    }


    /**
     * \overload OpenKalman::make_covariance
     * \brief For Eigen3: Make a writable, uninitialized Covariance, specifying the typed_index_descriptor.
     * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
     * \tparam Coefficients The coefficient types corresponding to the rows and columns.
     * \tparam Scalar The scalar type (integral or floating-point).
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, scalar_type Scalar = double>
#else
    template<typename Coefficients, typename Scalar = double, std::enable_if_t<
      typed_index_descriptor<Coefficients> and scalar_type<Scalar>, int> = 0>
#endif
    auto make_covariance()
    {
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
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
     * must equal dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>.
     */
#ifdef __cpp_concepts
    template<typed_index_descriptor Coefficients, TriangleType triangle_type = TriangleType::lower, scalar_type ... Args>
    requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>)
#else
    template<typename Coefficients, TriangleType triangle_type = TriangleType::lower, typename ... Args,
      std::enable_if_t<typed_index_descriptor<Coefficients> and (scalar_type<Args> and ...) and
        (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
        (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<Coefficients> * dimension_size_of_v<Coefficients>), int> = 0>
#endif
    auto make_square_root_covariance(const Args ... args)
    {
      using Scalar = std::decay_t<std::common_type_t<Args...>>;
      using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
      using T = typename MatrixTraits<Mat>::template TriangularMatrixFrom<triangle_type>;
      return SquareRootCovariance<Coefficients, T>(MatrixTraits<T>::make(static_cast<const Scalar>(args)...));
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
    template<TriangleType triangle_type = TriangleType::lower, scalar_type ... Args> requires
    (sizeof...(Args) > 0) and (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args)))
#else
    template<TriangleType triangle_type = TriangleType::lower, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) == OpenKalman::internal::constexpr_sqrt(sizeof...(Args)) *
        OpenKalman::internal::constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
    auto make_square_root_covariance(const Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      using Coefficients = OpenKalman::Dimensions<dim>;
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
    template<typed_index_descriptor Coefficients, TriangleType triangle_type = TriangleType::lower, scalar_type Scalar = double>
    requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
    template<typename Coefficients, TriangleType triangle_type = TriangleType::lower, typename Scalar = double,
      std::enable_if_t<typed_index_descriptor<Coefficients> and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and scalar_type<Scalar>, int> = 0>
#endif
    auto make_square_root_covariance()
    {
      using Mat = Eigen3::eigen_matrix_t<double, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>;
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
    typed_index_descriptor RowCoefficients,
    typed_index_descriptor ColumnCoefficients = RowCoefficients,
    typed_matrix_nestable NestedMatrix =
      Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>>
  requires
    (dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
    (dimension_size_of_v<ColumnCoefficients> == column_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_coefficients<RowCoefficients> == dynamic_rows<NestedMatrix>) and
    (dynamic_coefficients<ColumnCoefficients> == dynamic_columns<NestedMatrix>)
#else
  template<
    typename RowCoefficients,
    typename ColumnCoefficients = RowCoefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>>
#endif
  struct Matrix;


  // If the arguments are a sequence of scalars, deduce a single-column matrix.
#ifdef __cpp_concepts
  template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
  Matrix(const Args ...) -> Matrix<Dimensions<sizeof...(Args)>, Axis,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ---------- //
  //    Mean    //
  // ---------- //

  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the Coefficients.
#ifdef __cpp_concepts
  template<typed_index_descriptor RowCoefficients,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, 1>>
  requires (dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, 1>>
#endif
  struct Mean;


  /// If the arguments are a sequence of scalars, deduce a single-column mean with all Axis coefficients.
#ifdef __cpp_concepts
  template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
  Mean(const Args ...) -> Mean<Dimensions<sizeof...(Args)>,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ------------------- //
  //    EuclideanMean    //
  // ------------------- //

#ifdef __cpp_concepts
  template<
    typed_index_descriptor Coefficients,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, euclidean_dimension_size_of_v<Coefficients>, 1>>
  requires
    (euclidean_dimension_size_of_v<Coefficients> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, euclidean_dimension_size_of_v<Coefficients>, 1>>
#endif
  struct EuclideanMean;


  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
#ifdef __cpp_concepts
  template<scalar_type ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...), int> = 0>
#endif
  EuclideanMean(const Args ...) -> EuclideanMean<OpenKalman::Dimensions<sizeof...(Args)>,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ---------------- //
  //    Covariance    //
  // ---------------- //

#ifdef __cpp_concepts
  template<typed_index_descriptor Coefficients, covariance_nestable NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>>>
  requires (dimension_size_of_v<Coefficients> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>>>
#endif
  struct Covariance;


  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<scalar_type ... Args> requires (sizeof...(Args) > 0) and
    (sizeof...(Args) == constexpr_sqrt(sizeof...(Args)) * constexpr_sqrt(sizeof...(Args)))
#else
  template<typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == constexpr_sqrt(sizeof...(Args)) * constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
  explicit Covariance(const Args& ...) -> Covariance<Dimensions<constexpr_sqrt(sizeof...(Args))>,
  Eigen3::SelfAdjointMatrix<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    constexpr_sqrt(sizeof...(Args)), constexpr_sqrt(sizeof...(Args))>>>;


  // --------------------- //
  //  SquareRootCovariance //
  // --------------------- //

#ifdef __cpp_concepts
  template<typed_index_descriptor Coefficients, covariance_nestable NestedMatrix =
  Eigen3::SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>>>
    requires (dimension_size_of_v<Coefficients> == row_dimension_of_v<NestedMatrix>) and
      (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<Coefficients>, dimension_size_of_v<Coefficients>>>>
#endif
  struct SquareRootCovariance;


  /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<scalar_type ... Args> requires (sizeof...(Args) > 0) and
  (sizeof...(Args) == constexpr_sqrt(sizeof...(Args)) * constexpr_sqrt(sizeof...(Args)))
#else
  template<typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == constexpr_sqrt(sizeof...(Args)) * constexpr_sqrt(sizeof...(Args))), int> = 0>
#endif
  explicit SquareRootCovariance(const Args& ...) -> SquareRootCovariance<Dimensions<constexpr_sqrt(sizeof...(Args))>,
  Eigen3::TriangularMatrix<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    constexpr_sqrt(sizeof...(Args)), constexpr_sqrt(sizeof...(Args))>>>;


} // namespace OpenKalman

#endif // OPENKALMAN_FIRST_INTERFACE

#endif //OPENKALMAN_DEFAULT_OVERLOADS_HPP
