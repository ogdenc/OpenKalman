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

#ifndef OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP
#define OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP

#include <type_traits>

// ================================================ //
//   Type traits for Eigen interface matrix types   //
// ================================================ //

namespace OpenKalman
{

  namespace interface
  {
    // ----------------------- //
    //  IndexibleObjectTraits  //
    // ----------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T>
    struct IndexibleObjectTraits<T>
#else
    template<typename T>
    struct IndexibleObjectTraits<T, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      static constexpr std::size_t max_indices = 2;
      using scalar_type = scalar_type_of_t<pattern_matrix_of_t<T>>;
    };


    // ------------- //
    //  IndexTraits  //
    // ------------- //

    template<typename Nested, std::size_t N>
    struct IndexTraits<DiagonalMatrix<Nested>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<Nested, 0>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<0>(nested_matrix(arg));
      }
    };


    template<typename Nested, TriangleType t, std::size_t N>
    struct IndexTraits<SelfAdjointMatrix<Nested, t>, N>
    {
      static constexpr std::size_t dimension = dynamic_dimension<Nested, 0> ?
        index_dimension_of_v<Nested, 1> : index_dimension_of_v<Nested, 0>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return get_index_dimension_of<0>(nested_matrix(arg));
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
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return get_index_dimension_of<0>(nested_matrix(arg));
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
      static constexpr std::size_t dimension = N == 0 ? dimension_size_of_v<C> : index_dimension_of_v<Nested, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return arg.row_coefficients.dimension;
          else
            return get_index_dimension_of<N>(nested_matrix(arg));
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
      static constexpr std::size_t dimension = N == 0 ? euclidean_dimension_size_of_v<C> : index_dimension_of_v<Nested, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_dimension<Arg, N>)
        {
          if constexpr (N == 0)
            return arg.row_coefficients.euclidean_dimension;
          else
            return get_index_dimension_of<N>(nested_matrix(arg));
        }
        else
        {
          return dimension;
        }
      }
    };


    // ------------------------ //
    //  CoordinateSystemTraits  //
    // ------------------------ //

    template<typename C, typename Nested, std::size_t N>
    struct CoordinateSystemTraits<FromEuclideanExpr<C, Nested>, N>
    {
      using coordinate_system_types = C;

      template<typename Arg>
      static constexpr auto coordinate_system_types_at_runtime(Arg&& arg)
      {
        return std::get<N>(std::forward<Arg>(arg).my_dimensions);
      }
    };


    template<typename C, typename Nested, std::size_t N>
    struct CoordinateSystemTraits<ToEuclideanExpr<C, Nested>, N>
    {
      using coordinate_system_types = C;

      template<typename Arg>
      static constexpr auto coordinate_system_types_at_runtime(Arg&& arg)
      {
        return std::get<N>(std::forward<Arg>(arg).my_dimensions);
      }
    };


    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      static constexpr bool is_writable = false;


      template<typename...D>
      static auto make_default(D&&...d)
      {
        using Trait = EquivalentDenseWritableMatrix<std::decay_t<pattern_matrix_of_t<T>>, Scalar>;
        return Trait::make_default(std::forward<D>(d)...);
      }

      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<std::decay_t<pattern_matrix_of_t<T>>, Scalar>;
        return Trait::convert(std::forward<Arg>(arg));
      }

      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<std::decay_t<pattern_matrix_of_t<T>>, Scalar>;
        return Trait::to_native_matrix(std::forward<Arg>(arg));
      }

    };


#ifdef __cpp_concepts
    template<euclidean_expr T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<euclidean_expr<T>>>
#endif
    {
      static constexpr bool is_writable = false;


      template<typename...D>
      static auto make_default(D&&...d)
      {
        return make_default_dense_writable_matrix_like<pattern_matrix_of_t<T>, Scalar>(std::forward<D>(d)...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        if constexpr (has_untyped_index<Arg, 0>)
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
        return TriangularMatrix<decltype(n), triangle_type> {std::move(n)};
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
        return SelfAdjointMatrix<decltype(n), triangle_type> {std::move(n)};
      }
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct Dependencies<ToEuclideanExpr<TypedIndex, NestedMatrix>>
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
        return FromEuclideanExpr<TypedIndex, decltype(n)> {std::move(n)};
      }
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct Dependencies<FromEuclideanExpr<TypedIndex, NestedMatrix>>
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
        return ToEuclideanExpr<TypedIndex, decltype(n)> {std::move(n)};
      }
    };


    // ----------------------------- //
    //   SingleConstantMatrixTraits  //
    // ----------------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      template<typename...D>
      static constexpr auto make_zero_matrix(D&&...d)
      {
        return make_zero_matrix_like<pattern_matrix_of_t<T>, Scalar>(std::forward<D>(d)...);
      }


      template<auto...constant, typename...D>
      static constexpr auto make_constant_matrix(D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>, Scalar, constant...>(std::forward<D>(d)...);
      }


      template<typename S, typename...D>
      static constexpr auto make_runtime_constant(S&& s, D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>>(std::forward<S>(s), std::forward<D>(d)...);
      }
    };


#ifdef __cpp_concepts
    template<euclidean_expr T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar, std::enable_if_t<euclidean_expr<T>>>
#endif
    {
      template<typename...D>
      static constexpr auto make_zero_matrix(D&&...d)
      {
        return make_zero_matrix_like<pattern_matrix_of_t<T>, Scalar>(std::forward<D>(d)...);
      }


      template<auto...constant, typename...D>
      static constexpr auto make_constant_matrix(D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>, Scalar, constant...>(std::forward<D>(d)...);
      }


      template<typename S, typename...D>
      static constexpr auto make_runtime_constant(S&& s, D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>>(std::forward<S>(s), std::forward<D>(d)...);
      }
    };


    // ------------------------------------- //
    //   SingleConstantDiagonalMatrixTraits  //
    // ------------------------------------- //

#ifdef __cpp_concepts
    template<untyped_adapter T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar, std::enable_if_t<untyped_adapter<T>>>
#endif
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<pattern_matrix_of_t<T>, Scalar>(std::forward<D>(d));
      }
    };


#ifdef __cpp_concepts
    template<euclidean_expr T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar, std::enable_if_t<euclidean_expr<T>>>
#endif
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<pattern_matrix_of_t<T>, Scalar>(std::forward<D>(d));
      }
    };


    // ---------------- //
    //  SingleConstant  //
    // ---------------- //

#ifdef __cpp_concepts
    template<constant_matrix<Likelihood::maybe> NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix, Likelihood::maybe>>>
#endif
    {
      const DiagonalMatrix<NestedMatrix>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient{diagonal_of(xpr)};
      }
    };


#ifdef __cpp_concepts
    template<constant_diagonal_matrix<Likelihood::maybe> NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>>
#else
    template<typename NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix, Likelihood::maybe>>>
#endif
    {
      const TriangularMatrix<NestedMatrix, triangle_type>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_diagonal_coefficient{nested_matrix(xpr)};
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix<Likelihood::maybe> NestedMatrix>
    struct SingleConstant<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<TriangularMatrix<NestedMatrix, TriangleType::diagonal>, std::enable_if_t<
      constant_matrix<NestedMatrix, Likelihood::maybe>>>
#endif
    {
      const TriangularMatrix<NestedMatrix, TriangleType::diagonal>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient{nested_matrix(xpr)};
      }
    };


    template<typename NestedMatrix, TriangleType storage_type>
    struct SingleConstant<SelfAdjointMatrix<NestedMatrix, storage_type>>
    {
      const SelfAdjointMatrix<NestedMatrix, storage_type>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (storage_type != TriangleType::diagonal) return constant_coefficient{nested_matrix(xpr)};
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr (storage_type == TriangleType::diagonal and constant_matrix<NestedMatrix, Likelihood::maybe>)
          return constant_coefficient{nested_matrix(xpr)};
        else
          return constant_diagonal_coefficient{nested_matrix(xpr)};
      }
    };


#ifdef __cpp_concepts
    template<euclidean_index_descriptor TypedIndex, typename NestedMatrix>
    struct SingleConstant<ToEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct SingleConstant<ToEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<euclidean_index_descriptor<TypedIndex>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


#ifdef __cpp_concepts
    template<euclidean_index_descriptor TypedIndex, typename NestedMatrix>
    struct SingleConstant<FromEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct SingleConstant<FromEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<euclidean_index_descriptor<TypedIndex>>>
#endif
      : SingleConstant<std::decay_t<NestedMatrix>> {};


    // ---------------- //
    //  DiagonalTraits  //
    // ---------------- //

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


    template<typename TypedIndex, typename NestedMatrix>
    struct DiagonalTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>>
    {
      static constexpr bool is_diagonal = euclidean_index_descriptor<TypedIndex> and diagonal_matrix<NestedMatrix>;
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct DiagonalTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>>
    {
      static constexpr bool is_diagonal = euclidean_index_descriptor<TypedIndex> and diagonal_matrix<NestedMatrix>;
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
    template<euclidean_index_descriptor TypedIndex, triangular_matrix NestedMatrix>
    struct TriangularTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<
      triangular_matrix<NestedMatrix> and euclidean_index_descriptor<TypedIndex>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<NestedMatrix>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<euclidean_index_descriptor TypedIndex, triangular_matrix NestedMatrix>
    struct TriangularTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<
      triangular_matrix<NestedMatrix> and euclidean_index_descriptor<TypedIndex>>>
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
    template<euclidean_index_descriptor TypedIndex, hermitian_matrix NestedMatrix>
    struct HermitianTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct HermitianTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<
      hermitian_matrix<NestedMatrix> and euclidean_index_descriptor<TypedIndex>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


#ifdef __cpp_concepts
    template<euclidean_index_descriptor TypedIndex, hermitian_matrix NestedMatrix>
    struct HermitianTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>>
#else
    template<typename TypedIndex, typename NestedMatrix>
    struct HermitianTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>, std::enable_if_t<
      hermitian_matrix<NestedMatrix> and euclidean_index_descriptor<TypedIndex>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  } // namespace interface


  using namespace OpenKalman::internal;

  template<typename NestedMatrix>
  struct MatrixTraits<DiagonalMatrix<NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<NestedMatrix>>::template MatrixBaseFrom<Derived>;

    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = rows>
    using TriangularMatrixFrom = TriangularMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, triangle_type>;

    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = DiagonalMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, 1>>;


#ifdef __cpp_concepts
    template<column_vector Arg>
#else
    template<typename Arg, std::enable_if_t<column_vector<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      if constexpr (eigen_diagonal_expr<Arg>)
        return DiagonalMatrix<nested_matrix_of_t<Arg>> {std::forward<Arg>(arg)};
      else
        return DiagonalMatrix<Arg> {std::forward<Arg>(arg)};
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
      return make(MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...));
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
      using M = untyped_dense_writable_matrix_t<NestedMatrix, Scalar, rows, columns>;
      return make(make_self_contained(diagonal_of(MatrixTraits<std::decay_t<M>>::make(static_cast<const Scalar>(args)...))));
    }

  };


  template<typename NestedMatrix, TriangleType triangle_type>
  struct MatrixTraits<TriangularMatrix<NestedMatrix, triangle_type>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_dimension_of_v<NestedMatrix> :
      row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<NestedMatrix>>::template MatrixBaseFrom<Derived>;

    template<TriangleType t = triangle_type, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = rows>
    using TriangularMatrixFrom = TriangularMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, t>;

    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = DiagonalMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, 1>>;


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, typename Arg> requires square_matrix<NestedMatrix, Likelihood::maybe>
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<square_matrix<NestedMatrix, Likelihood::maybe>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return TriangularMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, diagonal_matrix Arg> requires eigen_self_adjoint_expr<Arg> or
      (eigen_triangular_expr<Arg> and triangle_type_of_v<Arg> == triangle_type_of_v<TriangularMatrixFrom<t>>)
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or
      (eigen_triangular_expr<Arg> and
        triangle_type_of<Arg>::value == triangle_type_of<TriangularMatrixFrom<t>>::value)), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return TriangularMatrix<nested_matrix_of_t<Arg>, t> {std::forward<Arg>(arg)};
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
      return make<t>(MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...));
    }

  };


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct MatrixTraits<SelfAdjointMatrix<NestedMatrix, storage_triangle>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dynamic_rows<NestedMatrix> ? column_dimension_of_v<NestedMatrix> :
          row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<NestedMatrix>>::template MatrixBaseFrom<Derived>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, t>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows>
    using TriangularMatrixFrom = TriangularMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, t>;

    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = DiagonalMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, 1>>;


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, typename Arg> requires
      (not diagonal_matrix<NestedMatrix> or not complex_number<scalar_type_of_t<NestedMatrix>>) and
      square_matrix<NestedMatrix, Likelihood::maybe>
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<
      (not diagonal_matrix<NestedMatrix> or not complex_number<scalar_type_of_t<NestedMatrix>>) and
      square_matrix<NestedMatrix, Likelihood::maybe>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SelfAdjointMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, diagonal_matrix Arg> requires
      eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SelfAdjointMatrix<nested_matrix_of_t<Arg>, t> {std::forward<Arg>(arg)};
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
      return make<t>(MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<FromEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = dimension_size_of_v<Coeffs>;
    static constexpr auto columns = column_dimension_of_v<NestedMatrix>;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<NestedMatrix>>::template MatrixBaseFrom<Derived>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows>
    using TriangularMatrixFrom = TriangularMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, triangle_type>;


    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = DiagonalMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, 1>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      (dynamic_index_descriptor<C> == dynamic_rows<Arg>) and
      (not fixed_index_descriptor<C> or euclidean_dimension_size_of_v<C> == row_dimension_of_v<Arg>) and
      (not dynamic_index_descriptor<C> or std::same_as<typename C::Scalar, scalar_type_of_t<Arg>>)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (dynamic_index_descriptor<C> == dynamic_rows<Arg>) and
      (not fixed_index_descriptor<C> or euclidean_dimension_size_of<C>::value == row_dimension_of<Arg>::value) and
      (not dynamic_index_descriptor<C> or std::is_same_v<typename C::Scalar, typename scalar_type_of<Arg>::type>), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return from_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == euclidean_dimension_size_of_v<Coeffs> * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<
      std::is_convertible<Args, Scalar>...> and (sizeof...(Args) == euclidean_dimension_size_of_v<Coeffs> * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<ToEuclideanExpr<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = euclidean_dimension_size_of_v<Coeffs>;
    static constexpr auto columns = column_dimension_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Dimensions<columns>;
    static_assert(dimension_size_of_v<Coeffs> == row_dimension_of_v<NestedMatrix>);


    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<NestedMatrix>>::template MatrixBaseFrom<Derived>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows>
    using TriangularMatrixFrom = TriangularMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, dim>, triangle_type>;


    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = DiagonalMatrix<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, dim, 1>>;


    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires (row_dimension_of_v<Arg> == dimension_size_of_v<C>)
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<
      (row_dimension_of<Arg>::value == dimension_size_of_v<C>), int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      return to_euclidean<C>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == dimension_size_of_v<Coeffs> * columns)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == dimension_size_of_v<Coeffs> * columns), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<std::decay_t<untyped_dense_writable_matrix_t<NestedMatrix, Scalar, rows, columns>>>::make(
        static_cast<const Scalar>(args)...));
    }

  };



  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

  template<typename N1, typename N2>
  struct is_modifiable_native<DiagonalMatrix<N1>, DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType storage_triangle, typename U> requires
    //(not hermitian_matrix<U>) or
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
    : std::bool_constant</*hermitian_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
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
      not equivalent_to<C, row_coefficient_types_of_t<U>>)) or
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
    : std::bool_constant<modifiable<NestedMatrix, dense_writable_matrix_t<NestedMatrix, scalar_type_of_t<NestedMatrix>>>/* and dimension_size_of_v<C> == row_dimension_of_v<U> and
      column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
      std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (euclidean_expr<U> and (from_euclidean_expr<U> or
      not modifiable<NestedMatrix, nested_matrix_of_t<U>> or
      not equivalent_to<C, row_coefficient_types_of_t<U>>)) or
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
    : std::bool_constant<modifiable<NestedMatrix, dense_writable_matrix_t<NestedMatrix, scalar_type_of_t<NestedMatrix>>>/* and euclidean_dimension_size_of_v<C> == row_dimension_of_v<U> and
      column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
      std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP
