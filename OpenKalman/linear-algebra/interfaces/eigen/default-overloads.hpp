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
   * must equal coordinates::dimension_of_v<RowCoefficients> * coordinates::dimension_of_v<ColumnCoefficients>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern RowCoefficients, fixed_pattern ColumnCoefficients = RowCoefficients, values::number ... Args>
  requires (sizeof...(Args) > 0) and (coordinates::dimension_of_v<RowCoefficients> * coordinates::dimension_of_v<ColumnCoefficients> == sizeof...(Args))
#else
  template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
  std::enable_if_t<(sizeof...(Args) > 0) and (values::number<Args> and ...) and
    (coordinates::dimension_of_v<RowCoefficients> * coordinates::dimension_of_v<ColumnCoefficients> == sizeof...(Args)), int> = 0>
#endif
  auto make_matrix(const Args...args)
  {
    using Scalar = std::common_type_t<Args...>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<RowCoefficients>, coordinates::dimension_of_v<ColumnCoefficients>>;
    return Matrix<RowCoefficients, ColumnCoefficients, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
  }


  /**
   * \overload
   * \brief For Eigen3: Make a one-column Matrix from a list of coefficients, with default Axis coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam Args A list of numerical coefficients (either integral or floating-point).
   */
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<
  (sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
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
  template<values::number Scalar, fixed_pattern RowCoefficients,
    fixed_pattern ColumnCoefficients = RowCoefficients>
#else
  template<typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
  std::enable_if_t<values::number<Scalar> and
    fixed_pattern<RowCoefficients> and fixed_pattern<ColumnCoefficients>, int> = 0>
#endif
  auto make_matrix()
  {
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<RowCoefficients>, coordinates::dimension_of_v<ColumnCoefficients>>;
    return Matrix<RowCoefficients, ColumnCoefficients, Mat>();
  }


  /**
   * \overload OpenKalman::make_mean
   * \brief For Eigen3: Make a Mean from a list of coefficients, specifying the row coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must be divisible by coordinates::dimension_of_v<StaticDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, values::number ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(Args) % coordinates::dimension_of_v<StaticDescriptor> == 0)
#else
  template<typename StaticDescriptor, typename ... Args, std::enable_if_t<fixed_pattern<StaticDescriptor> and
    (sizeof...(Args) > 0) and (values::number<Args> and ...) and
    (sizeof...(Args) % coordinates::dimension_of_v<StaticDescriptor> == 0), int> = 0>
#endif
  auto make_mean(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    constexpr std::size_t dim = coordinates::dimension_of_v<StaticDescriptor>;
    constexpr std::size_t cols = sizeof...(Args) / dim;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
    return Mean<StaticDescriptor, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
  }


  /**
   * \overload
   * \brief For Eigen3: Make a one-column Mean from a list of coefficients, with default Axis coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam Args A list of numerical coefficients (either integral or floating-point).
   */
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
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
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam cols The number of columns.
   */
#ifdef __cpp_concepts
  template<values::number Scalar, fixed_pattern StaticDescriptor, std::size_t cols = 1>
#else
  template<typename Scalar, typename StaticDescriptor, std::size_t cols = 1, std::enable_if_t<
  values::number<Scalar> and fixed_pattern<StaticDescriptor>, int> = 0>
#endif
  auto make_mean()
  {
    return Mean<StaticDescriptor, Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, cols>>();
  }


  /**
   * \overload OpenKalman::make_euclidean_mean
   * \brief For Eigen3: Make a EuclideanMean from a list of coefficients, specifying the row coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must be divisible by coordinates::stat_dimension_of_v<StaticDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, values::number ... Args> requires
  (sizeof...(Args) > 0) and (sizeof...(Args) % coordinates::stat_dimension_of_v<StaticDescriptor> == 0)
#else
  template<typename StaticDescriptor, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (values::number<Args> and ...) and
    (sizeof...(Args) % coordinates::stat_dimension_of_v<StaticDescriptor> == 0), int> = 0>
#endif
  auto make_euclidean_mean(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    constexpr std::size_t dim = coordinates::stat_dimension_of_v<StaticDescriptor>;
    constexpr std::size_t cols = sizeof...(Args) / dim;
    using Mat = Eigen3::eigen_matrix_t<Scalar, dim, cols>;
    return EuclideanMean<StaticDescriptor, Mat>(make_dense_object_from<Mat>(static_cast<const Scalar>(args)...));
  }


  /**
   * \overload
   * \brief For Eigen3: Make a one-column EuclideanMean from a list of coefficients, with default Axis coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam Args A list of numerical coefficients (either integral or floating-point).
   */
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
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
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam cols The number of columns.
   */
#ifdef __cpp_concepts
  template<values::number Scalar, fixed_pattern StaticDescriptor, std::size_t cols = 1>
#else
  template<typename Scalar, typename StaticDescriptor, std::size_t cols = 1, std::enable_if_t<
  values::number<Scalar> and fixed_pattern<StaticDescriptor>, int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return EuclideanMean<StaticDescriptor,  Eigen3::eigen_matrix_t<Scalar, coordinates::stat_dimension_of_v<StaticDescriptor>, cols>>();
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a Covariance, with nested triangular type, from a list of coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type, values::number ... Args> requires
  (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
  (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>)
#else
  template<typename StaticDescriptor, TriangleType triangle_type, typename ... Args, std::enable_if_t<
    fixed_pattern<StaticDescriptor> and (values::number<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    using T = TriangularAdapter<Mat, triangle_type>;
    using SA = HermitianAdapter<Mat, triangle_type == TriangleType::upper ? HermitianAdapterType::upper : HermitianAdapterType::lower>;
    return Covariance<StaticDescriptor, T>(SA {make_dense_object_from<Mat>(static_cast<const Scalar>(args)...)});
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a Covariance from a list of coefficients, specifying the coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, values::number ... Args> requires
  (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>)
#else
  template<typename StaticDescriptor, typename ... Args, std::enable_if_t<
    fixed_pattern<StaticDescriptor> and (values::number<Args> and ...) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    auto mat = make_dense_object_from<Mat>(static_cast<const Scalar>(args)...);
    using SA = HermitianAdapter<Mat>;
    return Covariance<StaticDescriptor, SA>(SA {mat});
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
  template<TriangleType triangle_type, values::number ... Args> requires
    (sizeof...(Args) > 0) and (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) * static_cast<std::size_t>(values::sqrt(sizeof...(Args))))
#else
  template<TriangleType triangle_type, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (values::number<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) * static_cast<std::size_t>(values::sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(values::sqrt(sizeof...(Args)));
    using StaticDescriptor = OpenKalman::Dimensions<dim>;
    return make_covariance<StaticDescriptor, triangle_type>(args...);
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a Covariance from a list of coefficients, with default Axis coefficients.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must be the square of an integer.
   */
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) * static_cast<std::size_t>(values::sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (values::number<Args> and ...) and
  (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) * static_cast<std::size_t>(values::sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(values::sqrt(sizeof...(Args)));
    using StaticDescriptor = OpenKalman::Dimensions<dim>;
    return make_covariance<StaticDescriptor>(args...);
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a writable, uninitialized Covariance with nested triangular matrix.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type, values::number Scalar = double>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
  template<typename StaticDescriptor, TriangleType triangle_type, typename Scalar = double, std::enable_if_t<
    fixed_pattern<StaticDescriptor> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    values::number<Scalar>, int> = 0>
#endif
  auto make_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    using T = TriangularAdapter<Mat, triangle_type>;
    return Covariance<StaticDescriptor, T>();
  }


  /**
   * \overload OpenKalman::make_covariance
   * \brief For Eigen3: Make a writable, uninitialized Covariance, specifying the fixed_pattern.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, values::number Scalar = double>
#else
  template<typename StaticDescriptor, typename Scalar = double, std::enable_if_t<
    fixed_pattern<StaticDescriptor> and values::number<Scalar>, int> = 0>
#endif
  auto make_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    using SA = HermitianAdapter<Mat>;
    return Covariance<StaticDescriptor, SA>();
  }


  /**
   * \overload OpenKalman::make_square_root_covariance
   * \brief For Eigen3: Make a SquareRootCovariance from a list of coefficients.
   * \details Only the coefficients in the associated upper or lower triangle are significant.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Args A list of numerical coefficients (either integral or floating-point). The number of coefficients
   * must equal coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type = TriangleType::lower, values::number ... Args>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>)
#else
  template<typename StaticDescriptor, TriangleType triangle_type = TriangleType::lower, typename ... Args,
    std::enable_if_t<fixed_pattern<StaticDescriptor> and (values::number<Args> and ...) and
      (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
      (sizeof...(Args) > 0) and (sizeof...(Args) == coordinates::dimension_of_v<StaticDescriptor> * coordinates::dimension_of_v<StaticDescriptor>), int> = 0>
#endif
  auto make_square_root_covariance(const Args ... args)
  {
    using Scalar = std::decay_t<std::common_type_t<Args...>>;
    using Mat = Eigen3::eigen_matrix_t<Scalar, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    auto mat = make_dense_object_from<Mat>(static_cast<const Scalar>(args)...);
    using Tri = TriangularAdapter<Mat, triangle_type>;
    auto tri = Tri {mat};
    return SquareRootCovariance<StaticDescriptor, Tri>(tri);
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
  template<TriangleType triangle_type = TriangleType::lower, values::number ... Args> requires
  (sizeof...(Args) > 0) and (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) * static_cast<std::size_t>(values::sqrt(sizeof...(Args))))
#else
  template<TriangleType triangle_type = TriangleType::lower, typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and (values::number<Args> and ...) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(values::sqrt(sizeof...(Args)))), int> = 0>
#endif
  auto make_square_root_covariance(const Args ... args)
  {
    constexpr auto dim = static_cast<std::size_t>(values::sqrt(sizeof...(Args)));
    using StaticDescriptor = OpenKalman::Dimensions<dim>;
    return make_square_root_covariance<StaticDescriptor, triangle_type>(args...);
  }


  /**
   * \overload OpenKalman::make_square_root_covariance
   * \brief For Eigen3: Make a writable, uninitialized SquareRootCovariance.
   * \note This function is imported into the OpenKalman namespace if Eigen3 is the first-included interface.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Scalar The scalar type (integral or floating-point).
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type = TriangleType::lower, values::number Scalar = double>
  requires (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper)
#else
  template<typename StaticDescriptor, TriangleType triangle_type = TriangleType::lower, typename Scalar = double,
    std::enable_if_t<fixed_pattern<StaticDescriptor> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and values::number<Scalar>, int> = 0>
#endif
  auto make_square_root_covariance()
  {
    using Mat = Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>;
    using T = TriangularAdapter<Mat, triangle_type>;
    return SquareRootCovariance<StaticDescriptor, T>();
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
    fixed_pattern RowCoefficients,
    fixed_pattern ColumnCoefficients = RowCoefficients,
    typed_matrix_nestable NestedMatrix =
      Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<RowCoefficients>, coordinates::dimension_of_v<ColumnCoefficients>>>
  requires
    (coordinates::dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (coordinates::dimension_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_pattern<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>) and
    (dynamic_pattern<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>)
#else
  template<
    typename RowCoefficients,
    typename ColumnCoefficients = RowCoefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<RowCoefficients>, coordinates::dimension_of_v<ColumnCoefficients>>>
#endif
  struct Matrix;


  // If the arguments are a sequence of scalars, deduce a single-column matrix.
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
#endif
  Matrix(const Args ...) -> Matrix<Dimensions<sizeof...(Args)>, Axis,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ---------- //
  //    Mean    //
  // ---------- //

  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the StaticDescriptor.
#ifdef __cpp_concepts
  template<fixed_pattern RowCoefficients,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<RowCoefficients>, 1>>
  requires (coordinates::dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<RowCoefficients>, 1>>
#endif
  struct Mean;


  /// If the arguments are a sequence of scalars, deduce a single-column mean with all Axis coefficients.
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
#endif
  Mean(const Args ...) -> Mean<Dimensions<sizeof...(Args)>,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ------------------- //
  //    EuclideanMean    //
  // ------------------- //

#ifdef __cpp_concepts
  template<
    fixed_pattern StaticDescriptor,
    typed_matrix_nestable NestedMatrix = Eigen3::eigen_matrix_t<double, coordinates::stat_dimension_of_v<StaticDescriptor>, 1>>
  requires
    (coordinates::stat_dimension_of_v<StaticDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename StaticDescriptor,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, coordinates::stat_dimension_of_v<StaticDescriptor>, 1>>
#endif
  struct EuclideanMean;


  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
#ifdef __cpp_concepts
  template<values::number ... Args> requires (sizeof...(Args) > 0)
#else
  template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and (values::number<Args> and ...), int> = 0>
#endif
  EuclideanMean(const Args ...) -> EuclideanMean<OpenKalman::Dimensions<sizeof...(Args)>,
    Eigen3::eigen_matrix_t<std::common_type_t<Args...>, sizeof...(Args), 1>>;


  // ---------------- //
  //    Covariance    //
  // ---------------- //

#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, covariance_nestable NestedMatrix =
    HermitianAdapter<Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>>>
  requires (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and values::number<scalar_type_of_t<NestedMatrix>>
#else
  template<typename StaticDescriptor, typename NestedMatrix =
    HermitianAdapter<Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>>>
#endif
  struct Covariance;


  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<values::number ... Args> requires (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(values::sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(values::number<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(values::sqrt(sizeof...(Args)))), int> = 0>
#endif
  explicit Covariance(const Args& ...) -> Covariance<Dimensions<static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>,
  HermitianAdapter<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    static_cast<std::size_t>(values::sqrt(sizeof...(Args))), static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>>>;


  // --------------------- //
  //  SquareRootCovariance //
  // --------------------- //

#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, covariance_nestable NestedMatrix =
  HermitianAdapter<Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>>>
    requires (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
      (not std::is_rvalue_reference_v<NestedMatrix>) and values::number<scalar_type_of_t<NestedMatrix>>
#else
  template<typename StaticDescriptor, typename NestedMatrix =
    HermitianAdapter<Eigen3::eigen_matrix_t<double, coordinates::dimension_of_v<StaticDescriptor>, coordinates::dimension_of_v<StaticDescriptor>>>>
#endif
  struct SquareRootCovariance;


  /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
#if defined(__cpp_concepts) && not defined(__clang__) // Because of compiler issue in at least clang version 10.0.0
  template<values::number ... Args> requires (sizeof...(Args) > 0) and
  (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) *
    static_cast<std::size_t>(values::sqrt(sizeof...(Args))))
#else
  template<typename ... Args, std::enable_if_t<(values::number<Args> and ...) and (sizeof...(Args) > 0) and
    (sizeof...(Args) == static_cast<std::size_t>(values::sqrt(sizeof...(Args))) *
      static_cast<std::size_t>(values::sqrt(sizeof...(Args)))), int> = 0>
#endif
  explicit SquareRootCovariance(const Args& ...) -> SquareRootCovariance<
    Dimensions<static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>,
    TriangularAdapter<Eigen3::eigen_matrix_t<std::common_type_t<Args...>,
    static_cast<std::size_t>(values::sqrt(sizeof...(Args))), static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>>>;


} // namespace OpenKalman

#endif // OPENKALMAN_FIRST_INTERFACE

#endif //OPENKALMAN_DEFAULT_OVERLOADS_HPP
