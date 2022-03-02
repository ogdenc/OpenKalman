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

    // -------------------- //
    //  StorageArrayTraits  //
    // -------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T>
    struct StorageArrayTraits<T>
#else
    template<typename T>
    struct StorageArrayTraits<T, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      static constexpr std::size_t max_indices = 2;
    };


    // ------------- //
    //  IndexTraits  //
    // ------------- //

    template<typename NestedMatrix, auto constant, std::size_t N>
    struct IndexTraits<ConstantMatrix<NestedMatrix, constant>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<NestedMatrix, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return arg.get_rows_at_runtime();
          else
            return arg.get_columns_at_runtime();
        }
        else
        {
          return dimension;
        }
      }
    };


    template<typename NestedMatrix, std::size_t N>
    struct IndexTraits<ZeroMatrix<NestedMatrix>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<NestedMatrix, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return arg.get_rows_at_runtime();
          else
            return arg.get_columns_at_runtime();
        }
        else
        {
          return dimension;
        }
      }
    };


    template<typename Nested, std::size_t N>
    struct IndexTraits<DiagonalMatrix<Nested>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<Nested, 0>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        return runtime_dimension_of<0>(nested_matrix(std::forward<Arg>(arg)));
      }
    };


    template<typename Nested, TriangleType t, std::size_t N>
    struct IndexTraits<SelfAdjointMatrix<Nested, t>, N>
    {
      static constexpr std::size_t dimension = dynamic_dimension<Nested, 0> ?
        index_dimension_of_v<Nested, 1> : index_dimension_of_v<Nested, 0>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return runtime_dimension_of<0>(nested_matrix(std::forward<Arg>(arg)));
          else
            return index_dimension_of_v<Nested, 1>;
        }
        else
        {
          return dimension;
        }
      }
    };


    template<typename Nested, TriangleType t, std::size_t N>
    struct IndexTraits<TriangularMatrix<Nested, t>, N>
    {
      static constexpr std::size_t dimension = dynamic_dimension<Nested, 0> ?
        index_dimension_of_v<Nested, 1> : index_dimension_of_v<Nested, 0>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return runtime_dimension_of<0>(nested_matrix(std::forward<Arg>(arg)));
          else
            return index_dimension_of_v<Nested, 1>;
        }
        else
        {
          return dimension;
        }
      }
    };


    template<typename C, typename Nested, std::size_t N>
    struct IndexTraits<FromEuclideanExpr<C, Nested>, N>
    {
      static constexpr std::size_t dimension = N == 0 ? C::dimension : index_dimension_of_v<Nested, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return std::forward<Arg>(arg).row_coefficients.dimension;
          else
            return runtime_dimension_of<N>(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          return dimension;
        }
      }
    };


    template<typename C, typename Nested, std::size_t N>
    struct IndexTraits<ToEuclideanExpr<C, Nested>, N>
    {
      static constexpr std::size_t dimension = N == 0 ? C::euclidean_dimension : index_dimension_of_v<Nested, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return std::forward<Arg>(arg).row_coefficients.euclidean_dimension;
          else
            return runtime_dimension_of<N>(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          return dimension;
        }
      }
    };


    // -------------- //
    //  ScalarTypeOf  //
    // -------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T>
    struct ScalarTypeOf<T>
#else
    template<typename T>
    struct ScalarTypeOf<T, std::enable_if_t<untyped_adapter<T>>>
#endif
      : ScalarTypeOf<pattern_matrix_of_t<T>> {};


    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar>
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {

      template<typename...runtime_dimensions>
      static auto make_default(runtime_dimensions...e)
      {
        return make_default_dense_writable_matrix_like<pattern_matrix_of_t<T>, rows, columns, Scalar>(e...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        if constexpr (euclidean_expr<T> and untyped_rows<T>)
        {
          return make_dense_writable_matrix_from(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          using M = std::decay_t<decltype(make_default_dense_writable_matrix_like(std::forward<Arg>(arg)))>;
          // \todo Create an alternate path in case (not std::is_constructible_v<M, Arg&&>)
          M m {std::forward<Arg>(arg)};
          return m;
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<pattern_matrix_of_t<Arg>>(std::forward<Arg>(arg));
      }

    };


    // --------------- //
    //   Dependencies  //
    // --------------- //

    template<typename NestedMatrix, auto constant>
    struct Dependencies<Eigen3::ConstantMatrix<NestedMatrix, constant>>
    {
      static constexpr bool has_runtime_parameters = index_dimension_of_v<NestedMatrix, 0> == dynamic_size or
        index_dimension_of_v<NestedMatrix, 1> == dynamic_size;

      using type = std::tuple<>;
    };


    template<typename NestedMatrix>
    struct Dependencies<Eigen3::ZeroMatrix<NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = index_dimension_of_v<NestedMatrix, 0> == dynamic_size or
        index_dimension_of_v<NestedMatrix, 1> == dynamic_size;

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


    // ----------------------------- //
    //   SingleConstantMatrixTraits  //
    // ----------------------------- //

    // This implements the default behavior of the make_zero_matrix interface function.
#ifdef __cpp_concepts
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    template<std::convertible_to<std::size_t>...runtime_dimensions> requires
      (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0))
    auto SingleConstantMatrixTraits<T, rows, columns, Scalar>::
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar, typename Enable>
    template<typename...runtime_dimensions, std::enable_if_t<
      sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int>>
    auto SingleConstantMatrixTraits<T, rows, columns, Scalar, Enable>::
#endif
    make_zero_matrix(runtime_dimensions...e)
    {
      using N = equivalent_dense_writable_matrix_t<T, rows, columns, Scalar>;
      return Eigen3::ZeroMatrix<N> {e...};
    }


    // This implements the default behavior of the make_zero_matrix interface function.
#ifdef __cpp_concepts
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    template<auto constant, std::convertible_to<std::size_t>...runtime_dimensions> requires
      (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0))
    auto SingleConstantMatrixTraits<T, rows, columns, Scalar>::
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar, typename Enable>
    template<auto constant, typename...runtime_dimensions, std::enable_if_t<
      sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int>>
    auto SingleConstantMatrixTraits<T, rows, columns, Scalar, Enable>::
#endif
    make_constant_matrix(runtime_dimensions...e)
    {
      using N = equivalent_dense_writable_matrix_t<T, rows, columns, Scalar>;
      return Eigen3::ConstantMatrix<N, constant> {e...};
    }


#ifdef __cpp_concepts
    template<untyped_adapter T, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<T, rows, columns, Scalar>
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<T, rows, columns, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        return make_zero_matrix_like<pattern_matrix_of_t<T>, rows, columns, Scalar>(e...);
      }


      template<auto constant, typename...runtime_dimensions>
      static auto make_constant_matrix(runtime_dimensions...e)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>, constant, rows, columns, Scalar>(e...);
      }
    };


    // ---------------- //
    //  SingleConstant  //
    // ---------------- //

    template<typename NestedMatrix, auto constant>
    struct SingleConstant<ConstantMatrix<NestedMatrix, constant>>
    {
      static constexpr auto value = constant;
    };


    template<typename NestedMatrix>
    struct SingleConstant<ZeroMatrix<NestedMatrix>>
    {
      static constexpr int value = 0;
    };


#ifdef __cpp_concepts
    template<constant_matrix NestedMatrix> requires zero_matrix<NestedMatrix> or
      (not dynamic_rows<DiagonalMatrix<NestedMatrix>> and row_dimension_of_v<DiagonalMatrix<NestedMatrix>> == 1)
    struct SingleConstant<DiagonalMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix> and
      (zero_matrix<NestedMatrix> or
        (not dynamic_rows<DiagonalMatrix<NestedMatrix>> and row_dimension_of<DiagonalMatrix<NestedMatrix>>::value == 1))>>
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


    // ------------------------------------- //
    //   SingleConstantDiagonalMatrixTraits  //
    // ------------------------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, dimension, Scalar>
#else
    template<typename T, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, dimension, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        return make_identity_matrix_like<pattern_matrix_of_t<T>, dimension, Scalar>(e...);
      }
    };


    // ------------------------ //
    //  SingleConstantDiagonal  //
    // ------------------------ //

#ifdef __cpp_concepts
    template<one_by_one_matrix NestedMatrix, auto constant>
    struct SingleConstantDiagonal<ConstantMatrix<NestedMatrix, constant>>
#else
    template<typename NestedMatrix, auto constant>
    struct SingleConstantDiagonal<ConstantMatrix<NestedMatrix, constant>, std::enable_if_t<
      one_by_one_matrix<NestedMatrix>>>
#endif
    {
      static constexpr auto value = constant;
    };


#ifdef __cpp_concepts
    template<square_matrix NestedMatrix>
    struct SingleConstantDiagonal<ZeroMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstantDiagonal<ZeroMatrix<NestedMatrix>, std::enable_if_t<square_matrix<NestedMatrix>>>
#endif
    {
      static constexpr int value = 0;
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


    // ---------------- //
    //  DiagonalTraits  //
    // ---------------- //

#ifdef __cpp_concepts
    template<square_matrix NestedMatrix, auto constant> requires
      one_by_one_matrix<NestedMatrix> or (are_within_tolerance(constant, 0))
    struct DiagonalTraits<ConstantMatrix<NestedMatrix, constant>>
#else
    template<typename NestedMatrix, auto constant>
    struct DiagonalTraits<ConstantMatrix<NestedMatrix, constant>, std::enable_if_t<square_matrix<NestedMatrix> and
      (one_by_one_matrix<NestedMatrix> or are_within_tolerance(constant, 0))>>
#endif
    {
      static constexpr bool is_diagonal = true;
    };


#ifdef __cpp_concepts
    template<square_matrix NestedMatrix>
    struct DiagonalTraits<ZeroMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct DiagonalTraits<ZeroMatrix<NestedMatrix>, std::enable_if_t<square_matrix<NestedMatrix>>>
#endif
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename NestedMatrix>
    struct DiagonalTraits<DiagonalMatrix<NestedMatrix>>
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename NestedMatrix, TriangleType triangle_type>
    struct DiagonalTraits<TriangularMatrix<NestedMatrix, triangle_type>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal or
        (upper_triangular_matrix<NestedMatrix> and triangle_type == TriangleType::lower) or
        (lower_triangular_matrix<NestedMatrix> and triangle_type == TriangleType::upper);
    };


    template<typename NestedMatrix, TriangleType storage_type>
    struct DiagonalTraits<SelfAdjointMatrix<NestedMatrix, storage_type>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<NestedMatrix> or storage_type == TriangleType::diagonal;
    };


    template<typename Coefficients, typename NestedMatrix>
    struct DiagonalTraits<ToEuclideanExpr<Coefficients, NestedMatrix>>
    {
      static constexpr bool is_diagonal = Coefficients::axes_only and diagonal_matrix<NestedMatrix>;
    };


    template<typename Coefficients, typename NestedMatrix>
    struct DiagonalTraits<FromEuclideanExpr<Coefficients, NestedMatrix>>
    {
      static constexpr bool is_diagonal = Coefficients::axes_only and diagonal_matrix<NestedMatrix>;
    };


    // ------------------ //
    //  TriangularTraits  //
    // ------------------ //

    template<typename NestedMatrix, TriangleType t>
    struct TriangularTraits<TriangularMatrix<NestedMatrix, t>>
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : t;
      static constexpr bool is_triangular_adapter = true;
    };


#ifdef __cpp_concepts
    template<typename Coefficients, triangular_matrix NestedMatrix> requires Coefficients::axes_only
    struct TriangularTraits<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct TriangularTraits<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      triangular_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<NestedMatrix>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<typename Coefficients, triangular_matrix NestedMatrix> requires Coefficients::axes_only
    struct TriangularTraits<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct TriangularTraits<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      triangular_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<NestedMatrix>;
      static constexpr bool is_triangular_adapter = false;
    };


    // ----------------- //
    //  HermitianTraits  //
    // ----------------- //

    template<typename NestedMatrix, TriangleType t>
    struct HermitianTraits<SelfAdjointMatrix<NestedMatrix, t>>
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : t;
    };


#ifdef __cpp_concepts
    template<typename Coefficients, self_adjoint_matrix NestedMatrix> requires Coefficients::axes_only
    struct HermitianTraits<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct HermitianTraits<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      self_adjoint_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = self_adjoint_triangle_type_of_v<NestedMatrix>;
    };


#ifdef __cpp_concepts
    template<typename Coefficients, self_adjoint_matrix NestedMatrix> requires Coefficients::axes_only
    struct HermitianTraits<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
    template<typename Coefficients, typename NestedMatrix>
    struct HermitianTraits<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      self_adjoint_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = self_adjoint_triangle_type_of_v<NestedMatrix>;
    };


  } // namespace interface


  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename NestedMatrix, auto constant>
  struct MatrixTraits<ConstantMatrix<NestedMatrix, constant>>
  {
  private:

    using Matrix = ConstantMatrix<NestedMatrix, constant>;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived, Matrix>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = TriangularMatrix<Matrix, triangle_type>;

    template<std::size_t dim = row_dimension_of_v<NestedMatrix>, typename S = scalar_type_of_t<NestedMatrix>>
    using DiagonalMatrixFrom = DiagonalMatrix<Eigen3::ConstantMatrix<
      equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>, constant>>;

  };


  template<typename NestedMatrix>
  struct MatrixTraits<ZeroMatrix<NestedMatrix>>
  {
  private:

    using Matrix = ZeroMatrix<NestedMatrix>;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived, Matrix>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;

    template<std::size_t dim = row_dimension_of_v<NestedMatrix>, typename S = scalar_type_of_t<NestedMatrix>>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::eigen_matrix_t<S, dim, 1>>;

  };


  template<typename NestedMatrix>
  struct MatrixTraits<Eigen3::DiagonalMatrix<NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived, Eigen3::DiagonalMatrix<NestedMatrix>>;

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
    template<std::convertible_to<Scalar> ... Args> requires (rows == dynamic_size) or (sizeof...(Args) == rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      ((rows == dynamic_size) or (sizeof...(Args) == rows)), int> = 0>
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
    requires (rows != dynamic_size) and (rows > 1) and (sizeof...(Args) == rows * rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (rows != dynamic_size) and (rows > 1) and (sizeof...(Args) == rows * rows), int> = 0>
#endif
    static auto
    make(const Args ... args)
    {
      using M = equivalent_dense_writable_matrix_t<NestedMatrix, rows, columns>;
      return make(make_self_contained(diagonal_of(MatrixTraits<M>::make(static_cast<const Scalar>(args)...))));
    }

  };


  template<typename NestedMatrix, TriangleType triangle_type>
  struct MatrixTraits<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_dimension_of_v<NestedMatrix> :
      row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived,
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

  };


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct MatrixTraits<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_dimension_of_v<NestedMatrix> :
          row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom =
      Eigen3::internal::Eigen3Base<Derived, Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>;

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

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = Coeffs::dimension;
    static constexpr auto columns = column_dimension_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::euclidean_dimension == row_dimension_of_v<NestedMatrix>);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived, Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and (row_dimension_of_v<Arg> == C::euclidean_dimension)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>) and
      (row_dimension_of<Arg>::value == C::euclidean_dimension), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return from_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::euclidean_dimension * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<
      std::is_convertible<Args, Scalar>...> and (sizeof...(Args) == Coeffs::euclidean_dimension * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Eigen3::ToEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = Coeffs::euclidean_dimension;
    static constexpr auto columns = column_dimension_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::dimension == row_dimension_of_v<NestedMatrix>);


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3Base<Derived, Eigen3::ToEuclideanExpr<Coeffs, NestedMatrix>>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, dim, S>, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<equivalent_dense_writable_matrix_t<NestedMatrix, dim, 1, S>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and (row_dimension_of_v<Arg> == C::dimension)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (Eigen3::eigen_matrix<Arg> or Eigen3::from_euclidean_expr<Arg>) and
      (row_dimension_of<Arg>::value == C::dimension), int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      using namespace Eigen3;
      return to_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == Coeffs::dimension * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == Coeffs::dimension * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<equivalent_dense_writable_matrix_t<NestedMatrix, rows>>::make(static_cast<const Scalar>(args)...));
    }

  };


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

  template<typename NestedMatrix, auto constant, typename U>
  struct is_modifiable_native<ConstantMatrix<NestedMatrix, constant>, U>
    : std::false_type {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<ZeroMatrix<NestedMatrix>, U>
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
    : std::bool_constant<modifiable<NestedMatrix, equivalent_dense_writable_matrix_t<NestedMatrix>>/* and C::dimension == row_dimension_of_v<U> and
      column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
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
    : std::bool_constant<modifiable<NestedMatrix, equivalent_dense_writable_matrix_t<NestedMatrix>>/* and C::euclidean_dimension == row_dimension_of_v<U> and
      column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
      std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP
