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
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients = RowCoefficients, scalar_type ... Args>
  requires (sizeof...(Args) > 0) and (dimension_size_of_v<RowCoefficients> * dimension_size_of_v<ColumnCoefficients> == sizeof...(Args))
#else
  template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
  std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (dimension_size_of_v<RowCoefficients> * dimension_size_of_v<ColumnCoefficients> == sizeof...(Args)), int> = 0>
#endif
  auto make_matrix(const Args...args)
  {
    using Scalar = std::common_type_t<Args...>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>;
    return Matrix<RowCoefficients, ColumnCoefficients, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
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
  template<scalar_type Scalar, fixed_vector_space_descriptor RowCoefficients,
    fixed_vector_space_descriptor ColumnCoefficients = RowCoefficients>
#else
  template<typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
  std::enable_if_t<scalar_type<Scalar> and
    fixed_vector_space_descriptor<RowCoefficients> and fixed_vector_space_descriptor<ColumnCoefficients>, int> = 0>
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
   * \tparam FixedDescriptor The coefficient types corresponding to the rows.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must be divisible by dimension_size_of_v<FixedDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, scalar_type ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(Args) % dimension_size_of_v<FixedDescriptor> == 0)
#else
  template<typename FixedDescriptor, typename ... Args, std::enable_if_t<fixed_vector_space_descriptor<FixedDescriptor> and
    (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (sizeof...(Args) % dimension_size_of_v<FixedDescriptor> == 0), int> = 0>
#endif
  auto make_mean(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    constexpr std::size_t dim = dimension_size_of_v<FixedDescriptor>;
    constexpr std::size_t cols = sizeof...(Args) / dim;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
    return Mean<FixedDescriptor, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
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
   * \tparam FixedDescriptor The coefficient types corresponding to the rows.
   * \tparam cols The number of columns.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, fixed_vector_space_descriptor FixedDescriptor, std::size_t cols = 1>
#else
  template<typename Scalar, typename FixedDescriptor, std::size_t cols = 1, std::enable_if_t<
  scalar_type<Scalar> and fixed_vector_space_descriptor<FixedDescriptor>, int> = 0>
#endif
  auto make_mean()
  {
    return Mean<FixedDescriptor, Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, cols>>();
  }


  /**
   * \overload OpenKalman::make_euclidean_mean
   * \brief For Eigen3: Make a EuclideanMean from a list of coefficients, specifying the row coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must be divisible by euclidean_dimension_size_of_v<FixedDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, scalar_type ... Args> requires
  (sizeof...(Args) > 0) and (sizeof...(Args) % euclidean_dimension_size_of_v<FixedDescriptor> == 0)
#else
  template<typename FixedDescriptor, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (sizeof...(Args) % euclidean_dimension_size_of_v<FixedDescriptor> == 0), int> = 0>
#endif
  auto make_euclidean_mean(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    constexpr std::size_t dim = euclidean_dimension_size_of_v<FixedDescriptor>;
    constexpr std::size_t cols = sizeof...(Args) / dim;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
    return EuclideanMean<FixedDescriptor, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
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
  auto make_euclidean_mean(const Args ... args)
  {
    return make_euclidean_mean<Dimensions<sizeof...(Args)>>(args...);
  }


  /**
   * \overload
   * \brief For Eigen3: Make a EuclideanMean based on a scalar type, row coefficients, and a number of columns.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam Scalar A scalar type (integral or floating-point).
   * \tparam FixedDescriptor The coefficient types corresponding to the rows.
   * \tparam cols The number of columns.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, fixed_vector_space_descriptor FixedDescriptor, std::size_t cols = 1>
#else
  template<typename Scalar, typename FixedDescriptor, std::size_t cols = 1, std::enable_if_t<
  scalar_type<Scalar> and fixed_vector_space_descriptor<FixedDescriptor>, int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return EuclideanMean<FixedDescriptor,  Eigen3::eigen_matrix_t<Scalar, euclidean_dimension_size_of_v<FixedDescriptor>, cols>>();
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a Covariance, with nested triangular type, from a list of coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, TriangleType triangle_type, scalar_type ... Args> requires
  (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
  (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>)
#else
  template<typename FixedDescriptor, TriangleType triangle_type, typename ... Args, std::enable_if_t<
    fixed_vector_space_descriptor<FixedDescriptor> and (scalar_type<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    using T = TriangularMatrix<Mat, triangle_type>;
    using SA = SelfAdjointMatrix<Mat, triangle_type == TriangleType::upper ? HermitianAdapterType::upper : HermitianAdapterType::lower>;
    return Covariance<FixedDescriptor, T>(SA {make_dense_object_from<Mat>(static_cast<const Scalar>(args)...)});
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a Covariance from a list of coefficients, specifying the coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, scalar_type ... Args> requires
  (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>)
#else
  template<typename FixedDescriptor, typename ... Args, std::enable_if_t<
    fixed_vector_space_descriptor<FixedDescriptor> and (scalar_type<Args> and ...) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    auto mat = make_dense_object_from<Mat>(static_cast<const Scalar>(args)...);
    using SA = SelfAdjointMatrix<Mat>;
    return Covariance<FixedDescriptor, SA>(SA {mat});
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
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) * static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))))
#else
  template<TriangleType triangle_type, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) * static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)));
    using FixedDescriptor = OpenKalman::Dimensions<dim>;
    return make_covariance<FixedDescriptor, triangle_type>(args...);
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
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) * static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
  (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) * static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)));
    using FixedDescriptor = OpenKalman::Dimensions<dim>;
    return make_covariance<FixedDescriptor>(args...);
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a writable, uninitialized Covariance with nested triangular matrix.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, TriangleType triangle_type, scalar_type Scalar = double>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
  template<typename FixedDescriptor, TriangleType triangle_type, typename Scalar = double, std::enable_if_t<
    fixed_vector_space_descriptor<FixedDescriptor> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    scalar_type<Scalar>, int> = 0>
#endif
  auto make_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    using T = TriangularMatrix<Mat, triangle_type>;
    return Covariance<FixedDescriptor, T>();
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a writable, uninitialized Covariance, specifying the fixed_vector_space_descriptor.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, scalar_type Scalar = double>
#else
  template<typename FixedDescriptor, typename Scalar = double, std::enable_if_t<
    fixed_vector_space_descriptor<FixedDescriptor> and scalar_type<Scalar>, int> = 0>
#endif
  auto make_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    using SA = SelfAdjointMatrix<Mat>;
    return Covariance<FixedDescriptor, SA>();
  }


  /**
   * \overload OpenKalman::make_square_root_covariance
   * \brief For Eigen3: Make a SquareRootCovariance from a list of coefficients.
   * \details Only the coefficients in the associated upper or lower triangle are significant.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, TriangleType triangle_type = TriangleType::lower, scalar_type ... Args>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>)
#else
  template<typename FixedDescriptor, TriangleType triangle_type = TriangleType::lower, typename ... Args,
    std::enable_if_t<fixed_vector_space_descriptor<FixedDescriptor> and (scalar_type<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == dimension_size_of_v<FixedDescriptor> * dimension_size_of_v<FixedDescriptor>), int> = 0>
#endif
  auto make_square_root_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    auto mat = make_dense_object_from<Mat>(static_cast<const Scalar>(args)...);
    using Tri = TriangularMatrix<Mat, triangle_type>;
    auto tri = Tri {mat};
    return SquareRootCovariance<FixedDescriptor, Tri>(tri);
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
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) * static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))))
#else
  template<TriangleType triangle_type = TriangleType::lower, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (scalar_type<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_square_root_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)));
    using FixedDescriptor = OpenKalman::Dimensions<dim>;
    return make_square_root_covariance<FixedDescriptor, triangle_type>(args...);
  }


  /**
   * \overload OpenKalman::make_square_root_covariance
   * \brief For Eigen3: Make a writable, uninitialized SquareRootCovariance.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam FixedDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, TriangleType triangle_type = TriangleType::lower, scalar_type Scalar = double>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
  template<typename FixedDescriptor, TriangleType triangle_type = TriangleType::lower, typename Scalar = double,
    std::enable_if_t<fixed_vector_space_descriptor<FixedDescriptor> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and scalar_type<Scalar>, int> = 0>
#endif
  auto make_square_root_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<double, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>;
    using T = TriangularMatrix<Mat, triangle_type>;
    return SquareRootCovariance<FixedDescriptor, T>();
  }


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

  // ------------ //
  //    Matrix    //
  // ------------ //

  // Default specialization for Eigen: the nested matrix will be an Eigen::Matrix of the appropriate size.
#ifdef __cpp_concepts
  template<
    fixed_vector_space_descriptor RowCoefficients,
    fixed_vector_space_descriptor ColumnCoefficients = RowCoefficients,
    typed_matrix_nestable NestedMatrix =
      Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, dimension_size_of_v<ColumnCoefficients>>>
  requires
    (dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (dimension_size_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_vector_space_descriptor<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>) and
    (dynamic_vector_space_descriptor<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>)
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

  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the FixedDescriptor.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, dimension_size_of_v<RowCoefficients>, 1>>
  requires (dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
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
    fixed_vector_space_descriptor FixedDescriptor,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, euclidean_dimension_size_of_v<FixedDescriptor>, 1>>
  requires
    (euclidean_dimension_size_of_v<FixedDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename FixedDescriptor,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, euclidean_dimension_size_of_v<FixedDescriptor>, 1>>
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
  template<fixed_vector_space_descriptor FixedDescriptor, covariance_nestable NestedMatrix =
    SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>>>
  requires (dimension_size_of_v<FixedDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename FixedDescriptor, typename NestedMatrix =
    SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>>>
#endif
  struct Covariance;


  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<scalar_type ... Args> requires (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))), int> = 0>
#endif
  explicit Covariance(const Args& ...) -> Covariance<Dimensions<static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))>,
  SelfAdjointMatrix<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))), static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))>>>;


  // --------------------- //
  //  SquareRootCovariance //
  // --------------------- //

#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, covariance_nestable NestedMatrix =
  SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>>>
    requires (dimension_size_of_v<FixedDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
      (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename FixedDescriptor, typename NestedMatrix =
    SelfAdjointMatrix<Eigen3::eigen_matrix_t<double, dimension_size_of_v<FixedDescriptor>, dimension_size_of_v<FixedDescriptor>>>>
#endif
  struct SquareRootCovariance;


  /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<scalar_type ... Args> requires (sizeof...(Args) > 0) and
  (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) *
    static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))), int> = 0>
#endif
  explicit SquareRootCovariance(const Args& ...) -> SquareRootCovariance<
    Dimensions<static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))>,
    TriangularMatrix<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args))), static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))>>>;


} // namespace OpenKalman

#endif // OPENKALMAN_FIRST_INTERFACE

#endif //OPENKALMAN_DEFAULT_OVERLOADS_HPP
