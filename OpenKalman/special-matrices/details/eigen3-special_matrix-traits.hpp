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
 * \brief Type traits as applied to special matrix classes in OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP

#include <type_traits>

// ================================================ //
//   Type traits for Eigen interface matrix types   //
// ================================================ //

namespace OpenKalman
{
  using namespace OpenKalman::Eigen3;
  using namespace OpenKalman::internal;


  namespace interface
  {

    // ------------- //
    //  RowExtentOf  //
    // ------------- //

    template<typename Scalar, auto constant, std::size_t row_extent, std::size_t column_extent>
    struct RowExtentOf<ConstantMatrix<Scalar, constant, row_extent, column_extent>>
      : std::integral_constant<std::size_t, row_extent>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ConstantMatrix<Scalar, constant, row_extent, column_extent>>);
        if constexpr (dynamic_rows<Arg>)
          return arg.rows();
        else
          return row_extent;
      }
    };


    template<typename Scalar, std::size_t row_extent, std::size_t column_extent>
    struct RowExtentOf<ZeroMatrix<Scalar, row_extent, column_extent>>
      : std::integral_constant<std::size_t, row_extent>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ZeroMatrix<Scalar, row_extent, column_extent>>);
        if constexpr (dynamic_rows<Arg>)
          return arg.rows();
        else
          return row_extent;
      }
    };


    template<typename N>
    struct RowExtentOf<DiagonalMatrix<N>> : RowExtentOf<std::decay_t<N>>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, DiagonalMatrix<N>>);
        return row_count(nested_matrix(std::forward<Arg>(arg)));
      }
    };


    template<typename N, TriangleType t>
    struct RowExtentOf<SelfAdjointMatrix<N, t>>
      : std::integral_constant<std::size_t, dynamic_rows<N> ? column_extent_of_v<N> : row_extent_of_v<N>>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, SelfAdjointMatrix<N, t>>);
        if constexpr (dynamic_rows<N>)
        {
          if constexpr (dynamic_columns<N>)
            return row_count(nested_matrix(std::forward<Arg>(arg)));
          else
            return column_extent_of_v<N>;
        }
        else
        {
          return RowExtentOf::value;
        }
      }
   };


    template<typename N, TriangleType t>
    struct RowExtentOf<TriangularMatrix<N, t>>
      : std::integral_constant<std::size_t, dynamic_rows<N> ? column_extent_of_v<N> : row_extent_of_v<N>>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, TriangularMatrix<N, t>>);
        if constexpr (dynamic_rows<N>)
        {
          if constexpr (dynamic_columns<N>)
            return row_count(nested_matrix(std::forward<Arg>(arg)));
          else
            return column_extent_of_v<N>;
        }
        else
        {
          return RowExtentOf::value;
        }
      }
   };


    template<typename C, typename N>
    struct RowExtentOf<FromEuclideanExpr<C, N>> : std::integral_constant<std::size_t, C::dimensions>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, FromEuclideanExpr<C, N>>);
        if constexpr (dynamic_rows<Arg>)
          return std::forward<Arg>(arg).row_coefficients.dimensions;
        else
          return C::dimensions;
      }
    };


    template<typename C, typename N>
    struct RowExtentOf<ToEuclideanExpr<C, N>> : std::integral_constant<std::size_t, C::euclidean_dimensions>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ToEuclideanExpr<C, N>>);
        if constexpr (dynamic_rows<Arg>)
          return std::forward<Arg>(arg).row_coefficients.euclidean_dimensions;
        else
          return C::euclidean_dimensions;
      }
    };


    // ---------------- //
    //  ColumnExtentOf  //
    // ---------------- //

    template<typename Scalar, auto constant, std::size_t row_extent, std::size_t column_extent>
    struct ColumnExtentOf<ConstantMatrix<Scalar, constant, row_extent, column_extent>>
      : std::integral_constant<std::size_t, column_extent>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ConstantMatrix<Scalar, constant, row_extent, column_extent>>);
        if constexpr (dynamic_columns<Arg>)
          return arg.cols();
        else
          return column_extent;
      }
    };


    template<typename Scalar, std::size_t row_extent, std::size_t column_extent>
    struct ColumnExtentOf<ZeroMatrix<Scalar, row_extent, column_extent>>
      : std::integral_constant<std::size_t, column_extent>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ZeroMatrix<Scalar, row_extent, column_extent>>);
        if constexpr (dynamic_columns<Arg>)
          return arg.cols();
        else
          return column_extent;
      }
    };


    template<typename N>
    struct ColumnExtentOf<DiagonalMatrix<N>> : std::integral_constant<std::size_t, row_extent_of_v<N>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, DiagonalMatrix<N>>);
        return row_count(std::forward<Arg>(arg));
      }
   };


    template<typename N, TriangleType t>
    struct ColumnExtentOf<SelfAdjointMatrix<N, t>>
      : std::integral_constant<std::size_t, dynamic_columns<N> ? row_extent_of_v<N> : column_extent_of_v<N>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, SelfAdjointMatrix<N, t>>);
        if constexpr (dynamic_columns<N>)
        {
          if constexpr (dynamic_rows<N>)
            return column_count(nested_matrix(std::forward<Arg>(arg)));
          else
            return row_extent_of_v<N>;
        }
        else
        {
          return ColumnExtentOf::value;
        }
      }
    };


    template<typename N, TriangleType t>
    struct ColumnExtentOf<TriangularMatrix<N, t>>
      : std::integral_constant<std::size_t, dynamic_columns<N> ? row_extent_of_v<N> : column_extent_of_v<N>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, TriangularMatrix<N, t>>);
        if constexpr (dynamic_columns<N>)
        {
          if constexpr (dynamic_rows<N>)
            return column_count(nested_matrix(std::forward<Arg>(arg)));
          else
            return row_extent_of_v<N>;
        }
        else
        {
          return ColumnExtentOf::value;
        }
      }
    };


    template<typename C, typename N>
    struct ColumnExtentOf<FromEuclideanExpr<C, N>> : ColumnExtentOf<std::decay_t<N>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, FromEuclideanExpr<C, N>>);
        return column_count(nested_matrix(std::forward<Arg>(arg)));
      }
    };


    template<typename C, typename N>
    struct ColumnExtentOf<ToEuclideanExpr<C, N>> : ColumnExtentOf<std::decay_t<N>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, ToEuclideanExpr<C, N>>);
        return column_count(nested_matrix(std::forward<Arg>(arg)));
      }
    };


    // -------------- //
    //  ScalarTypeOf  //
    // -------------- //

    template<typename Scalar, auto constant, std::size_t row_extent, std::size_t column_extent>
    struct ScalarTypeOf<ConstantMatrix<Scalar, constant, row_extent, column_extent>>
    {
      using type = Scalar;
    };


    template<typename Scalar, std::size_t row_extent, std::size_t column_extent>
    struct ScalarTypeOf<ZeroMatrix<Scalar, row_extent, column_extent>>
    {
      using type = Scalar;
    };


#ifdef __cpp_concepts
    template<typename T> requires eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T> or
      euclidean_expr<T>
    struct ScalarTypeOf<T>
#else
    template<typename T>
    struct ScalarTypeOf<T, std::enable_if_t<
      eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T> or euclidean_expr<T>>>
#endif
      : ScalarTypeOf<nested_matrix_of_t<T>> {};


    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type> requires
      eigen_constant_expr<T> or eigen_zero_expr<T>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<
      eigen_constant_expr<T> or eigen_zero_expr<T>>>
#endif
      : EquivalentDenseWritableMatrix<Eigen3::eigen_matrix_t<scalar_type, row_extent, column_extent>,
        row_extent, column_extent, scalar_type>
    {
#ifdef __cpp_concepts
      template<typename Arg> requires
        (row_extent_of_v<Arg> == dynamic_extent or row_extent_of_v<Arg> == row_extent) and
        (column_extent_of_v<Arg> == dynamic_extent or column_extent_of_v<Arg> == column_extent) and
        std::convertible_to<scalar_type_of_t<Arg>, scalar_type>
 #else
      template<typename Arg, std::enable_if_t<
        (row_extent_of<Arg>::value == dynamic_extent or row_extent_of<Arg>::value == row_extent) and
        (column_extent_of_v<Arg> == dynamic_extent or column_extent_of<Arg>::value == column_extent) and
        std::is_convertible_v<typename scalar_type_of<Arg>::type, scalar_type>, int> = 0>
 #endif
      static decltype(auto) convert(Arg&& arg)
      {
        using M = Eigen3::eigen_matrix_t<scalar_type, row_extent, column_extent>;
        using Base = EquivalentDenseWritableMatrix<M, row_extent, column_extent, scalar_type>;
        return Base::convert(std::forward<Arg>(arg));
      }
    };


#ifdef __cpp_concepts
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type> requires
      eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<
      eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
      : EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, row_extent, column_extent, scalar_type>
    {
#ifdef __cpp_concepts
      template<typename Arg> requires
        (row_extent_of_v<Arg> == dynamic_extent or row_extent_of_v<Arg> == row_extent) and
        (column_extent_of_v<Arg> == dynamic_extent or column_extent_of_v<Arg> == column_extent) and
        std::convertible_to<scalar_type_of_t<Arg>, scalar_type>
 #else
      template<typename Arg, std::enable_if_t<
        (row_extent_of<Arg>::value == dynamic_extent or row_extent_of<Arg>::value == row_extent) and
        (column_extent_of<Arg>::value == dynamic_extent or column_extent_of<Arg>::value == column_extent) and
        std::is_convertible_v<typename scalar_type_of<Arg>::type, scalar_type>, int> = 0>
 #endif
      static decltype(auto) convert(Arg&& arg)
      {
        using Base = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, row_extent, column_extent, scalar_type>;
        return Base::convert(std::forward<Arg>(arg));
      }
    };


#ifdef __cpp_concepts
    template<euclidean_expr T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<euclidean_expr<T>>>
#endif
      : EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, row_extent, column_extent, scalar_type>
    {
#ifdef __cpp_concepts
      template<typename Arg> requires
        (row_extent_of_v<Arg> == dynamic_extent or row_extent_of_v<Arg> == row_extent) and
        (column_extent_of_v<Arg> == dynamic_extent or column_extent_of_v<Arg> == column_extent) and
        std::convertible_to<scalar_type_of_t<Arg>, scalar_type>
 #else
      template<typename Arg, std::enable_if_t<
        (row_extent_of<Arg>::value == dynamic_extent or row_extent_of<Arg>::value == row_extent) and
        (column_extent_of<Arg>::value == dynamic_extent or column_extent_of<Arg>::value == column_extent) and
        std::is_convertible_v<typename scalar_type_of<Arg>::type, scalar_type>, int> = 0>
 #endif
      static decltype(auto) convert(Arg&& arg)
      {
        using Base = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, row_extent, column_extent, scalar_type>;

        if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
        {
          return Base::convert(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          // \todo Make this more general to apply to interfaces other than Eigen
          using Base = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, row_extent, column_extent, scalar_type>;
          return typename Base::type {std::forward<Arg>(arg)};
        }
      }
    };


    // ----------------- //
    //   Dependencies  //
    // ----------------- //

    template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
    struct Dependencies<Eigen3::ConstantMatrix<Scalar, constant, rows, columns>>
    {
      static constexpr bool has_runtime_parameters = rows == dynamic_extent or columns == dynamic_extent;
      using type = std::tuple<>;
    };


    template<typename Scalar, std::size_t rows, std::size_t columns>
    struct Dependencies<Eigen3::ZeroMatrix<Scalar, rows, columns>>
    {
      static constexpr bool has_runtime_parameters = rows == dynamic_extent or columns == dynamic_extent;
      using type = std::tuple<>;
    };


    template<typename ColumnVector>
    struct Dependencies<DiagonalMatrix<ColumnVector>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<ColumnVector>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return DiagonalMatrix {make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)))};
      }
    };


    template<typename NestedMatrix, TriangleType triangle_type>
    struct Dependencies<TriangularMatrix<NestedMatrix, triangle_type>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return Eigen3::TriangularMatrix<decltype(n), triangle_type> {std::move(n)};
      }
    };


    template<typename NestedMatrix, TriangleType triangle_type>
    struct Dependencies<SelfAdjointMatrix<NestedMatrix, triangle_type>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return Eigen3::SelfAdjointMatrix<decltype(n), triangle_type> {std::move(n)};
      }
    };


    template<typename Coefficients, typename NestedMatrix>
    struct Dependencies<ToEuclideanExpr<Coefficients, NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return Eigen3::FromEuclideanExpr<Coefficients, decltype(n)> {std::move(n)};
      }
    };


    template<typename Coefficients, typename NestedMatrix>
    struct Dependencies<FromEuclideanExpr<Coefficients, NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return Eigen3::ToEuclideanExpr<Coefficients, decltype(n)> {std::move(n)};
      }
    };


    // ---------------- //
    //  SingleConstant  //
    // ---------------- //

    template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
    struct SingleConstant<ConstantMatrix<Scalar, constant, rows, columns>>
    {
      static constexpr auto value = constant;
    };


    template<typename Scalar, std::size_t rows, std::size_t columns>
    struct SingleConstant<ZeroMatrix<Scalar, rows, columns>>
    {
      static constexpr Scalar value = 0;
    };


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix> requires zero_matrix<NestedMatrix> or
      (not dynamic_rows<DiagonalMatrix<NestedMatrix>> and row_extent_of_v<DiagonalMatrix<NestedMatrix>> == 1)
    struct SingleConstant<DiagonalMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix> and
      (zero_matrix<NestedMatrix> or
        (not dynamic_rows<DiagonalMatrix<NestedMatrix>> and row_extent_of<DiagonalMatrix<NestedMatrix>>::value == 1))>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<zero_matrix NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>>
#else
    template<typename NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<zero_matrix<NestedMatrix>>>
#endif
    {
      static constexpr scalar_type_of_t<NestedMatrix> value = 0;
    };


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix, TriangleType storage_type> requires
      (not complex_number<scalar_type_of_t<NestedMatrix>>) or (std::imag(constant_coefficient_v<NestedMatrix>) == 0)
    struct SingleConstant<SelfAdjointMatrix<NestedMatrix, storage_type>>
#else
    template<typename NestedMatrix, TriangleType storage_type>
    struct SingleConstant<SelfAdjointMatrix<NestedMatrix, storage_type>, std::enable_if_t<
      constant_matrix<NestedMatrix> and (not complex_number<scalar_type_of<NestedMatrix>::type>)>>
      : SingleConstant<std::decay_t<NestedMatrix>> {};

    template<typename NestedMatrix, TriangleType storage_type>
    struct SingleConstant<SelfAdjointMatrix<NestedMatrix, storage_type>, std::enable_if_t<
      constant_matrix<NestedMatrix> and std::imag(constant_coefficient<NestedMatrix>::value) == 0>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<typename Coefficients, constant_matrix NestedMatrix> requires Coefficients::axes_only
    struct SingleConstant<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct SingleConstant<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      Coefficients::axes_only and constant_matrix<NestedMatrix>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<typename Coefficients, constant_matrix NestedMatrix> requires Coefficients::axes_only
    struct SingleConstant<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct SingleConstant<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      Coefficients::axes_only and constant_matrix<NestedMatrix>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


    // ------------------------ //
    //  SingleConstantDiagonal  //
    // ------------------------ //

    template<typename Scalar, auto constant>
    struct SingleConstantDiagonal<ConstantMatrix<Scalar, constant, 1, 1>>
    {
      static constexpr auto value = constant;
    };


#ifdef __cpp_concepts
    template<typename Scalar, std::size_t dim> requires (dim != dynamic_extent)
    struct SingleConstantDiagonal<ZeroMatrix<Scalar, dim, dim>>
#else
    template<typename Scalar, std::size_t dim>
    struct SingleConstantDiagonal<ZeroMatrix<Scalar, dim, dim>, std::enable_if_t<dim != dynamic_extent>>
#endif
    {
      static constexpr Scalar value = 0;
    };


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix>
    struct SingleConstantDiagonal<DiagonalMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstantDiagonal<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<constant_diagonal_matrix NestedMatrix, TriangleType triangle_type>
    struct SingleConstantDiagonal<TriangularMatrix<NestedMatrix, triangle_type>>
#else
    template<typename NestedMatrix, TriangleType triangle_type>
    struct SingleConstantDiagonal<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix>>>
#endif
      : SingleConstantDiagonal<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix>
    struct SingleConstantDiagonal<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
#else
    template<typename NestedMatrix, TriangleType triangle_type>
    struct SingleConstantDiagonal<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
      constant_matrix<NestedMatrix>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<constant_diagonal_matrix NestedMatrix, TriangleType storage_type>
    struct SingleConstantDiagonal<SelfAdjointMatrix<NestedMatrix, storage_type>>
#else
    template<typename NestedMatrix, TriangleType storage_type>
    struct SingleConstantDiagonal<SelfAdjointMatrix<NestedMatrix, storage_type>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix>>>
#endif
      : SingleConstantDiagonal<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix>
    struct SingleConstantDiagonal<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
#else
    template<typename NestedMatrix>
    struct SingleConstantDiagonal<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>, std::enable_if_t<
      constant_matrix<NestedMatrix>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<typename Coefficients, constant_diagonal_matrix NestedMatrix> requires Coefficients::axes_only
    struct SingleConstantDiagonal<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct SingleConstantDiagonal<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
        constant_diagonal_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
      : SingleConstantDiagonal<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<typename Coefficients, constant_diagonal_matrix NestedMatrix> requires Coefficients::axes_only
    struct SingleConstantDiagonal<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct SingleConstantDiagonal<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
      : SingleConstantDiagonal<std::decay_t<NestedMatrix>> {};

  } // namespace interface


  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
  struct MatrixTraits<Eigen3::ConstantMatrix<Scalar, constant, rows, columns>>
  {
  private:

    using Matrix = Eigen3::ConstantMatrix<Scalar, constant, rows, columns>;

  public:

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::ConstantMatrix<S, constant, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived>;


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {static_cast<std::size_t>(args)...};
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
    requires (sizeof...(Args) >= (rows == dynamic_extent and columns == dynamic_extent ? 1 : 0)) and
      (sizeof...(Args) <= 1)
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) >= (rows == dynamic_extent and columns == dynamic_extent ? 1 : 0)) and
      (sizeof...(Args) <= 1), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      constexpr std::size_t r = sizeof...(Args) == 0 ? (rows == dynamic_extent ? columns : rows) : dynamic_extent;
      return DiagonalMatrix { Eigen3::ConstantMatrix<Scalar, 1, r, 1> {args...} };
    }

  };


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct MatrixTraits<Eigen3::ZeroMatrix<Scalar, rows, columns>>
  {
  private:

    using Matrix = Eigen3::ZeroMatrix<Scalar, rows, columns>;

  public:

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::eigen_matrix_t<S, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived>;


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {static_cast<std::size_t>(args)...};
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
    requires (sizeof...(Args) >= (rows == dynamic_extent and columns == dynamic_extent ? 1 : 0)) and
      (sizeof...(Args) <= 1)
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) >= (rows == dynamic_extent and columns == dynamic_extent ? 1 : 0)) and
      (sizeof...(Args) <= 1), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      constexpr std::size_t r = sizeof...(Args) == 0 ? (rows == dynamic_extent ? columns : rows) : dynamic_extent;
      return DiagonalMatrix { Eigen3::ConstantMatrix<Scalar, 1, r, 1> {args...} };
    }

  };


  template<typename NestedMatrix>
  struct MatrixTraits<Eigen3::DiagonalMatrix<NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::DiagonalMatrix<NestedMatrix>>;

    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, triangle_type>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


#ifdef __cpp_concepts
    template<column_vector Arg>
#else
    template<typename Arg, std::enable_if_t<column_vector<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_diagonal_expr<Arg>)
        return Eigen3::DiagonalMatrix<nested_matrix_of_t<Arg>> {std::forward<Arg>(arg)};
      else
        return Eigen3::DiagonalMatrix<Arg> {std::forward<Arg>(arg)};
    }


    /** Make diagonal matrix using a list of coefficients defining the diagonal.
     * The size of the list must match the number of diagonal coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args>
    requires (rows == dynamic_extent) or (sizeof...(Args) == rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      ((rows == dynamic_extent) or (sizeof...(Args) == rows)), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


    /** Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
     * The size of the list must match the number of coefficients in the matrix (diagonal and non-diagonal).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args>
    requires (rows != dynamic_extent) and (rows > 1) and (sizeof...(Args) == rows * rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (rows != dynamic_extent) and (rows > 1) and (sizeof...(Args) == rows * rows), int> = 0>
#endif
    static auto
    make(const Args ... args)
    {
      using M = equivalent_dense_writable_matrix_t<NestedMatrix, rows, columns>;
      return make(make_self_contained(Eigen3::diagonal_of(MatrixTraits<M>::make(static_cast<const Scalar>(args)...))));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows, columns>>::zero(
        static_cast<std::size_t>(args)..., static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows, columns>>::identity(args...);
    }

  };


  template<typename NestedMatrix, TriangleType triangle_type>
  struct MatrixTraits<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_extent_of_v<NestedMatrix> :
      row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived,
      Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, t>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, typename Arg> requires
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::TriangularMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, diagonal_matrix Arg> requires Eigen3::eigen_self_adjoint_expr<Arg> or
      (Eigen3::eigen_triangular_expr<Arg> and triangle_type_of_v<Arg> == triangle_type_of_v<TriangularMatrixFrom<t>>)
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (Eigen3::eigen_self_adjoint_expr<Arg> or
      (Eigen3::eigen_triangular_expr<Arg> and
        triangle_type_of<Arg>::value == triangle_type_of<TriangularMatrixFrom<t>>::value)), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::TriangularMatrix<nested_matrix_of_t<Arg>, t> {std::forward<Arg>(arg)};
    }


    /// Make triangular matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts

    template<TriangleType t = triangle_type, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = triangle_type, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make(const Args...args)
    {
      return make<t>(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::identity(args...);
    }

  };


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct MatrixTraits<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_extent_of_v<NestedMatrix> :
          row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom =
      Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, t>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, t>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>)
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, diagonal_matrix Arg> requires
      Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<nested_matrix_of_t<Arg>, t> {std::forward<Arg>(arg)};
    }


    /// Make self-adjoint matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = storage_triangle, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make<t>(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0) + (columns == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::identity(args...);
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = Coeffs::dimensions;
    static constexpr auto columns = column_extent_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::euclidean_dimensions == row_extent_of_v<NestedMatrix>);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and (row_extent_of_v<Arg> == C::euclidean_dimensions)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and
      (row_extent_of<Arg>::value == C::euclidean_dimensions), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return from_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::euclidean_dimensions * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<
      std::is_convertible<Args, Scalar>...> and (sizeof...(Args) == Coeffs::euclidean_dimensions * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::identity(args...);
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Eigen3::ToEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = Coeffs::euclidean_dimensions;
    static constexpr auto columns = column_extent_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::dimensions == row_extent_of_v<NestedMatrix>);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::ToEuclideanExpr<Coeffs, NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and (row_extent_of_v<Arg> == C::dimensions)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and
      (row_extent_of<Arg>::value == C::dimensions), int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return to_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::dimensions * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == Coeffs::dimensions * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::identity(args...);
    }

  };


  // ----------------------------- //
  //  is_upper_self_adjoint_matrix  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::upper>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  // ----------------------------- //
  //  is_lower_self_adjoint_matrix  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::lower>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  // ------------------------ //
  //  is_covariance_nestable  //
  // ------------------------ //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref covariance.
#ifdef __cpp_concepts
  template<typename T> requires
    eigen_constant_expr<T> or
    eigen_zero_expr<T> or
    eigen_diagonal_expr<T> or
    eigen_self_adjoint_expr<T> or
    eigen_triangular_expr<T>
  struct is_covariance_nestable<T>
#else
  template<typename T>
  struct is_covariance_nestable<T, std::enable_if_t<
    Eigen3::eigen_constant_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T>>>
#endif
  : std::true_type {};


  // -------------------------- //
  //  is_typed_matrix_nestable  //
  // -------------------------- //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref typed_matrix.
#ifdef __cpp_concepts
  template<typename T>
  requires eigen_zero_expr<T> or eigen_constant_expr<T>
  struct is_typed_matrix_nestable<T>
#else
  template<typename T>
  struct is_typed_matrix_nestable<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T>>>
#endif
  : std::true_type {};


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_diagonal_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent) and (dim == 1 or constant == 0)> {};


  template<typename Scalar, std::size_t dim>
  struct is_diagonal_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<dim != dynamic_extent> {};


  template<typename NestedMatrix>
  struct is_diagonal_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_diagonal_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal or
      (upper_triangular_matrix<NestedMatrix> and triangle_type == TriangleType::lower) or
      (lower_triangular_matrix<NestedMatrix> and triangle_type == TriangleType::upper)> {};


  template<typename NestedMatrix, TriangleType storage_type>
  struct is_diagonal_matrix<SelfAdjointMatrix<NestedMatrix, storage_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or storage_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  // ------------------------------ //
  //  is_lower_self_adjoint_matrix  //
  // ------------------------------ //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_lower_self_adjoint_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent) and not complex_number<decltype(constant)>> {};


  template<typename Scalar, std::size_t dim>
  struct is_lower_self_adjoint_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent)> {};


  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<DiagonalMatrix<NestedMatrix>>
    : std::bool_constant<not complex_number<scalar_type_of_t<NestedMatrix>>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<storage_triangle != TriangleType::upper> {};


  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::bool_constant<not complex_number<scalar_type_of_t<NestedMatrix>>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_self_adjoint_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_self_adjoint_matrix<NestedMatrix>> {};


  // ------------------------------ //
  //  is_upper_self_adjoint_matrix  //
  // ------------------------------ //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_upper_self_adjoint_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent) and not complex_number<decltype(constant)>> {};


  template<typename Scalar, std::size_t dim>
  struct is_upper_self_adjoint_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent)> {};


  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<DiagonalMatrix<NestedMatrix>>
    : std::bool_constant<not complex_number<scalar_type_of_t<NestedMatrix>>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<storage_triangle != TriangleType::lower> {};


  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::bool_constant<not complex_number<scalar_type_of_t<NestedMatrix>>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_self_adjoint_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_self_adjoint_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_lower_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_lower_triangular_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent) and (dim == 1 or constant == 0)> {};


  template<typename Scalar, std::size_t dim>
  struct is_lower_triangular_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim != dynamic_extent)> {};


  template<typename NestedMatrix>
  struct is_lower_triangular_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_lower_triangular_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<lower_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::lower or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_upper_triangular_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<constant == 0 or dim == 1> {};


  template<typename Scalar, std::size_t dim>
  struct is_upper_triangular_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<dim != dynamic_extent> {};


  template<typename NestedMatrix>
  struct is_upper_triangular_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_upper_triangular_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<upper_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::upper or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  // ------------- //
  //  is_writable  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T>
  requires (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T> or
    to_euclidean_expr<T> or from_euclidean_expr<T>) and
    writable<nested_matrix_of_t<T>>
  struct is_writable<T> : std::true_type {};
#else
  template<typename T>
    struct is_writable<T, std::enable_if_t<
      (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T> or
      to_euclidean_expr<T> or from_euclidean_expr<T>) and
      writable<nested_matrix_of_t<T>>>> : std::true_type {};
#endif


  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

  template<typename Scalar, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<ZeroMatrix<Scalar, rows, columns>, U>
    : std::false_type {};


  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<ConstantMatrix<Scalar, constant, rows, columns>, U>
    : std::false_type {};


  template<typename N1, typename N2>
  struct is_modifiable_native<DiagonalMatrix<N1>, DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType storage_triangle, typename U> requires
    //(not self_adjoint_matrix<U>) or
    //(storage_triangle == TriangleType::diagonal and not diagonal_matrix<U>) or
    (eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, nested_matrix_of_t<U>>) or
    (not eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, storage_triangle>, U> : std::false_type {};
#else
  template<typename N1, TriangleType t1, typename N2, TriangleType t2>
  struct is_modifiable_native<SelfAdjointMatrix<N1, t1>, SelfAdjointMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, TriangleType t, typename U>
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*self_adjoint_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
      */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type, typename U> requires
    //(not triangular_matrix<U>) or
    //(triangle_type == TriangleType::diagonal and not diagonal_matrix<U>) or
    (eigen_triangular_expr<U> and not modifiable<NestedMatrix, nested_matrix_of_t<U>>) or
    (not eigen_triangular_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<TriangularMatrix<NestedMatrix, triangle_type>, U> : std::false_type {};
#else
  template<typename N1, TriangleType t1, typename N2, TriangleType t2>
  struct is_modifiable_native<TriangularMatrix<N1, t1>, TriangularMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, TriangleType t, typename U>
  struct is_modifiable_native<TriangularMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*triangular_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
      */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (euclidean_expr<U> and (to_euclidean_expr<U> or
      not modifiable<NestedMatrix, nested_matrix_of_t<U>> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
    (not euclidean_expr<U> and not modifiable<NestedMatrix, ToEuclideanExpr<C, std::decay_t<U>>>)
  struct is_modifiable_native<FromEuclideanExpr<C, NestedMatrix>, U>
    : std::false_type {};
#else
  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<FromEuclideanExpr<C1, N1>, FromEuclideanExpr<C2, N2>>
    : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<FromEuclideanExpr<C1, N1>, ToEuclideanExpr<C2, N2>>
    : std::false_type {};

  template<typename C, typename NestedMatrix, typename U>
  struct is_modifiable_native<FromEuclideanExpr<C, NestedMatrix>, U>
    : std::bool_constant<modifiable<NestedMatrix, equivalent_dense_writable_matrix_t<NestedMatrix>>/* and C::dimensions == row_extent_of_v<U> and
      column_extent_of_v<NestedMatrix> == column_extent_of_v<U> and
      std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (euclidean_expr<U> and (from_euclidean_expr<U> or
      not modifiable<NestedMatrix, nested_matrix_of_t<U>> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
    (not euclidean_expr<U> and not modifiable<NestedMatrix, FromEuclideanExpr<C, std::decay_t<U>>>)
  struct is_modifiable_native<ToEuclideanExpr<C, NestedMatrix>, U>
  : std::false_type {};
#else
  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<ToEuclideanExpr<C1, N1>, ToEuclideanExpr<C2, N2>>
    : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<ToEuclideanExpr<C1, N1>, FromEuclideanExpr<C2, N2>>
    : std::false_type {};

  template<typename C, typename NestedMatrix, typename U>
  struct is_modifiable_native<ToEuclideanExpr<C, NestedMatrix>, U, std::void_t<FromEuclideanExpr<C, std::decay_t<U>>>>
    : std::bool_constant<modifiable<NestedMatrix, equivalent_dense_writable_matrix_t<NestedMatrix>>/* and C::euclidean_dimensions == row_extent_of_v<U> and
      column_extent_of_v<NestedMatrix> == column_extent_of_v<U> and
      std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP
