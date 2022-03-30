/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIXTRAITS_HPP
#define OPENKALMAN_MATRIXTRAITS_HPP

namespace OpenKalman
{

  namespace interface
  {

    // -------------------- //
    //  StorageArrayTraits  //
    // -------------------- //

#ifdef __cpp_concepts
    template<typed_adapter T>
    struct StorageArrayTraits<T>
#else
    template<typename T>
    struct StorageArrayTraits<T, std::enable_if_t<typed_adapter<T>>>
#endif
    {
      static constexpr std::size_t max_indices = 2;
    };


    // ------------- //
    //  IndexTraits  //
    // ------------- //

#ifdef __cpp_concepts
    template<typed_matrix T, std::size_t N>
    struct IndexTraits<T, N>
#else
    template<typename T, std::size_t N>
    struct IndexTraits<T, N, std::enable_if_t<typed_matrix<T>>>
#endif
    {
      static constexpr std::size_t dimension = N == 0 ? row_coefficient_types_of_t<T>::dimension :
        column_coefficient_types_of_t<T>::dimension;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(Arg&& arg)
      {
        if constexpr (dynamic_dimension<T, N>)
          return runtime_dimension_of<N>(nested_matrix(std::forward<Arg>(arg)));
        else
          return dimension;
      }
    };


#ifdef __cpp_concepts
    template<covariance T, std::size_t N>
    struct IndexTraits<T, N>
#else
    template<typename T, std::size_t N>
    struct IndexTraits<T, N, std::enable_if_t<covariance<T>>>
#endif
    {
      static constexpr std::size_t dimension = MatrixTraits<T>::Coefficients::dimension;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        using Nested = nested_matrix_of_t<T>;

        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return runtime_dimension_of<0>(nested_matrix(arg));
          else
            return index_dimension_of_v<Nested, 1>;
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

    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix, std::size_t N>
    struct CoordinateSystemTraits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>, N>
    {
      using coordinate_system_types = std::conditional_t<N == 0, RowCoeffs, ColCoeffs>;
    };


    template<typename Coeffs, typename NestedMatrix>
    struct CoordinateSystemTraits<Mean<Coeffs, NestedMatrix>, 0>
    {
      using coordinate_system_types = Coeffs;
    };


    template<typename Coeffs, typename NestedMatrix>
    struct CoordinateSystemTraits<EuclideanMean<Coeffs, NestedMatrix>, 0>
    {
      using coordinate_system_types = Coeffs;
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t N>
    struct CoordinateSystemTraits<Covariance<Coeffs, NestedMatrix>, N>
    {
      using coordinate_system_types = Coeffs;
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t N>
    struct CoordinateSystemTraits<SquareRootCovariance<Coeffs, NestedMatrix>, N>
    {
      using coordinate_system_types = Coeffs;
    };


    // -------------- //
    //  ScalarTypeOf  //
    // -------------- //

#ifdef __cpp_concepts
    template<typed_adapter T>
    struct ScalarTypeOf<T>
#else
    template<typename T>
    struct ScalarTypeOf<T, std::enable_if_t<typed_adapter<T>>>
#endif
    {
      using type = scalar_type_of_t<pattern_matrix_of_t<T>>;
    };


    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<covariance T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar>
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar, std::enable_if_t<covariance<T>>>
#endif
    {

      template<typename...D>
      static auto make_default(D&&...d)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, rows, columns, Scalar>;
        return Trait::make_default(std::forward<D>(d)...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, rows, columns, Scalar>;
        return Trait::convert(OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg)));
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix(
          OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg))(std::forward<Arg>(arg)));
      }

    };


#ifdef __cpp_concepts
    template<typed_matrix T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar>
#else
    template<typename T, std::size_t rows, std::size_t columns, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, rows, columns, Scalar, std::enable_if_t<typed_matrix<T>>>
#endif
    {

      template<typename...D>
      static auto make_default(D&&...d)
      {
        using Trait = EquivalentDenseWritableMatrix<pattern_matrix_of_t<T>, rows, columns, Scalar>;
        return Trait::make_default(std::forward<D>(d)...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, rows, columns, Scalar>;
        return Trait::convert(nested_matrix(std::forward<Arg>(arg)));
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix(nested_matrix(std::forward<Arg>(arg)));
      }

    };


    // --------------- //
    //   Dependencies  //
    // --------------- //

    template<typename Coeffs, typename NestedMatrix>
    struct Dependencies<Covariance<Coeffs, NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (self_adjoint_matrix<NestedMatrix>)
        {
          return std::forward<Arg>(arg).get_self_adjoint_nested_matrix();
        }
        else
        {
          return std::forward<Arg>(arg).get_triangular_nested_matrix();
        }
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return Covariance<Coeffs, decltype(n)> {std::move(n)};
      }
    };


    template<typename Coeffs, typename NestedMatrix>
    struct Dependencies<SquareRootCovariance<Coeffs, NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (self_adjoint_matrix<NestedMatrix>)
        {
          return std::forward<Arg>(arg).get_self_adjoint_nested_matrix();
        }
        else
        {
          return std::forward<Arg>(arg).get_triangular_nested_matrix();
        }
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
        return SquareRootCovariance<Coeffs, decltype(n)> {std::move(n)};
      }
    };


    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix>
    struct Dependencies<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>>
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
        auto n = make_self_contained(std::forward<Arg>(arg).nested_matrix());
        return Matrix<RowCoeffs, ColCoeffs, decltype(n)> {std::move(n)};
      }
    };


    template<typename Coeffs, typename NestedMatrix>
    struct Dependencies<Mean<Coeffs, NestedMatrix>>
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
        auto n = make_self_contained(std::forward<Arg>(arg).nested_matrix());
        return Mean<Coeffs, decltype(n)> {std::move(n)};
      }
    };


    template<typename Coeffs, typename NestedMatrix>
    struct Dependencies<EuclideanMean<Coeffs, NestedMatrix>>
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
        auto n = make_self_contained(std::forward<Arg>(arg).nested_matrix());
        return EuclideanMean<Coeffs, decltype(n)> {std::move(n)};
      }
    };


    // ----------------------------- //
    //   SingleConstantMatrixTraits  //
    // ----------------------------- //

    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix,
      std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>, rows, columns, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0));

        auto n = make_zero_matrix_like<NestedMatrix, rows, columns, Scalar>(e...);
        return Matrix<RowCoeffs, ColCoeffs, std::decay_t<decltype(n)>>(std::move(n)) //< \todo use make_matrix
      }
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<Mean<Coeffs, NestedMatrix>, rows, columns, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0));

        auto n = wrap_angles<Coeffs>(make_zero_matrix_like<NestedMatrix, rows, columns, Scalar>(e...));
        return Mean<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_mean
      }
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<EuclideanMean<Coeffs, NestedMatrix>, rows, columns, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0));

        auto n = make_zero_matrix_like<NestedMatrix, rows, columns, Scalar>(e...);
        return EuclideanMean<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_mean
      }
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<Covariance<Coeffs, NestedMatrix>, rows, columns, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0));

        auto n = make_zero_matrix_like<NestedMatrix, rows, columns, Scalar>(e...);
        return Covariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_covariance
      }
    };


    template<typename Coeffs, typename NestedMatrix, std::size_t rows, std::size_t columns, typename Scalar>
    struct SingleConstantMatrixTraits<SquareRootCovariance<Coeffs, NestedMatrix>, rows, columns, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_zero_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0));

        auto n = make_zero_matrix_like<NestedMatrix, rows, columns, Scalar>(e...);
        return SquareRootCovariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)) //< \todo use make_square_root_covariance
      }
    };


    // ---------------- //
    //  SingleConstant  //
    // ---------------- //

    /*
     * A typed matrix or covariance is a constant matrix if its nested matrix is a constant matrix.
     * In the case of a triangular_covariance, the nested matrix must also be a zero_matrix.
     */
#ifdef __cpp_concepts
    template<typename T> requires
      ((typed_matrix<T> or self_adjoint_covariance<T>) and constant_matrix<nested_matrix_of_t<T>>) or
      (triangular_covariance<T> and zero_matrix<nested_matrix_of_t<T>>)
    struct SingleConstant<T>
#else
    template<typename T>
    struct SingleConstant<T, std::enable_if_t<
      ((typed_matrix<T> or self_adjoint_covariance<T>) and constant_matrix<nested_matrix_of<T>::type>) or
      (triangular_covariance<T> and zero_matrix<nested_matrix_of<T>::type>)>>>
#endif
      : SingleConstant<std::decay_t<nested_matrix_of_t<T>>> {};


      // ------------------------------------- //
      //   SingleConstantDiagonalMatrixTraits  //
      // ------------------------------------- //

    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>, dimension, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (dimension == dynamic_size ? 1 : 0));

        auto n = make_identity_matrix_like<NestedMatrix, dimension, Scalar>(e...);
        return Matrix<RowCoeffs, ColCoeffs, std::decay_t<decltype(n)>>(std::move(n)) //< \ todo use make_matrix
      }

    };


    template<typename Coeffs, typename NestedMatrix, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Mean<Coeffs, NestedMatrix>, dimension, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (dimension == dynamic_size ? 1 : 0));

        auto n = make_identity_matrix_like<NestedMatrix, dimension, Scalar>(e...);
        return Matrix<Coeffs, Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_matrix
      }

    };


    template<typename Coeffs, typename NestedMatrix, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<EuclideanMean<Coeffs, NestedMatrix>, dimension, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (dimension == dynamic_size ? 1 : 0));

        auto n = make_identity_matrix_like<NestedMatrix, dimension, Scalar>(e...);
        return Matrix<Coeffs, Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_matrix
      }

    };


    template<typename Coeffs, typename NestedMatrix, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Covariance<Coeffs, NestedMatrix>, dimension, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(e) == (dimension == dynamic_size ? 1 : 0));

        auto n = make_identity_matrix_like<NestedMatrix, dimension, Scalar>(e...);
        return Covariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_covariance
      }

    };


    template<typename Coeffs, typename NestedMatrix, std::size_t dimension, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<SquareRootCovariance<Coeffs, NestedMatrix>, dimension, Scalar>
    {
      template<typename...runtime_dimensions>
      static auto make_identity_matrix(runtime_dimensions...e)
      {
        static_assert((std::is_convertible_v<runtime_dimensions, std::size_t> and ...));
        static_assert(sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0));

        auto n = make_identity_matrix_like<NestedMatrix, dimension, Scalar>(e...);
        return SquareRootCovariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_square_root_covariance
      }

    };


    // ------------------------ //
    //  SingleConstantDiagonal  //
    // ------------------------ //

    /*
     * A typed_matrix or covariance is a constant_diagonal_matrix if its nested matrix is a constant_diagonal_matrix.
     */
#ifdef __cpp_concepts
    template<typename T> requires typed_matrix<T> or covariance<T>
    struct SingleConstantDiagonal<T>
#else
    template<typename T>
    struct SingleConstantDiagonal<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
#endif
      : SingleConstantDiagonal<std::decay_t<nested_matrix_of_t<T>>> {};


    // ---------------- //
    //  DiagonalTraits  //
    // ---------------- //

    /*
     * A covariance is a diagonal matrix if its nested matrix is diagonal
     */
#ifdef __cpp_concepts
    template<covariance T>
    struct DiagonalTraits<T>
#else
    template<typename T>
    struct DiagonalTraits<T, std::enable_if_t<covariance<T>>>
#endif
    {
      static constexpr bool is_diagonal = diagonal_matrix<nested_matrix_of_t<T>>;
    };


    /*
     * A typed matrix is diagonal if its nested matrix is diagonal and its row and column coefficients are equivalent.
     */
#ifdef __cpp_concepts
    template<typed_matrix T>
    struct DiagonalTraits<T>
#else
    template<typename T>
    struct DiagonalTraits<T, std::enable_if_t<typed_matrix<T>>>
#endif
    {
      static constexpr bool is_diagonal = diagonal_matrix<nested_matrix_of_t<T>> and
        equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>;
    };


    // ------------------ //
    //  TriangularTraits  //
    // ------------------ //

#ifdef __cpp_concepts
    template<triangular_covariance T> requires triangular_matrix<nested_matrix_of_t<T>>
    struct TriangularTraits<T>
#else
    template<typename T>
    struct TriangularTraits<T, std::enable_if_t<triangular_covariance<T> and
      triangular_matrix<nested_matrix_of<T>::type>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<nested_matrix_of_t<T>>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<triangular_covariance T> requires self_adjoint_matrix<nested_matrix_of_t<T>>
    struct TriangularTraits<T>
#else
    template<typename T>
    struct TriangularTraits<T, std::enable_if_t<triangular_covariance<T> and
      self_adjoint_matrix<nested_matrix_of<T>::type>>>
#endif
    {
      static constexpr TriangleType triangle_type = self_adjoint_triangle_type_of_v<nested_matrix_of_t<T>>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<typed_matrix T> requires triangular_matrix<nested_matrix_of_t<T>> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>
    struct TriangularTraits<T>
#else
    template<typename T>
    struct TriangularTraits<T, std::enable_if_t<
      typed_matrix<T> and triangular_matrix<nested_matrix_of<T>::type> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<nested_matrix_of_t<T>>;
      static constexpr bool is_triangular_adapter = false;
    };


    // ----------------- //
    //  HermitianTraits  //
    // ----------------- //

#ifdef __cpp_concepts
    template<self_adjoint_covariance T> requires self_adjoint_matrix<nested_matrix_of_t<T>>
    struct HermitianTraits<T>
#else
    template<typename T>
    struct HermitianTraits<T, std::enable_if_t<self_adjoint_covariance<T> and
      self_adjoint_matrix<nested_matrix_of<T>::type>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = self_adjoint_triangle_type_of_v<nested_matrix_of_t<T>>;
    };


#ifdef __cpp_concepts
    template<self_adjoint_covariance T> requires triangular_matrix<nested_matrix_of_t<T>>
    struct HermitianTraits<T>
#else
    template<typename T>
    struct HermitianTraits<T, std::enable_if_t<self_adjoint_covariance<T> and
      triangular_matrix<nested_matrix_of<T>::type>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = triangle_type_of_v<nested_matrix_of_t<T>>;
    };


#ifdef __cpp_concepts
    template<typed_matrix T> requires self_adjoint_matrix<nested_matrix_of_t<T>> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>
    struct HermitianTraits<T>
#else
    template<typename T>
    struct HermitianTraits<T, std::enable_if_t<
      typed_matrix<T> and self_adjoint_matrix<nested_matrix_of<T>::type> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>>>
#endif
      : self_adjoint_triangle_type_of<nested_matrix_of_t<T>> {};

  } // namespace interface


  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename RowCoefficients, typename ColCoefficients, typename NestedMatrix>
  struct MatrixTraits<Matrix<RowCoefficients, ColCoefficients, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr std::size_t rows = row_dimension_of_v<NestedMatrix>;
    static constexpr std::size_t columns = column_dimension_of_v<NestedMatrix>;
    static_assert(RowCoefficients::dimension == rows);
    static_assert(ColCoefficients::dimension == columns);

  public:

#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, coefficients CC = ColumnCoefficients, typed_matrix_nestable Arg>
    requires (row_dimension_of_v<Arg> == RC::dimension) and (column_dimension_of_v<Arg> == CC::dimension)
#else
    template<typename RC = RowCoefficients, typename CC = ColumnCoefficients, typename Arg, std::enable_if_t<
      coefficients<RC> and coefficients<CC> and typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == RC::dimension) and (column_dimension_of<Arg>::value == CC::dimension), int> = 0>
#endif
    static auto make(Arg&& arg)
    {
      return Matrix<RC, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  template<typename RowCoefficients, typename NestedMatrix>
  struct MatrixTraits<Mean<RowCoefficients, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = column_dimension_of_v<NestedMatrix>;

  public:

    static_assert(RowCoefficients::dimension == rows);

    /// Make from a typed_matrix_nestable. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      (std::is_void_v<CC> or equivalent_to<CC, Axes<column_dimension_of_v<Arg>>>) and
      (row_dimension_of_v<Arg> == RC::dimension)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      (std::is_void_v<CC> or equivalent_to<CC, Axes<column_dimension_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_dimension_of<Arg>::value == RC::dimension), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg)); using B = decltype(b);
      return Mean<RC, std::decay_t<B>>(std::forward<B>(b));
    }

  };


  template<typename RowCoefficients, typename NestedMatrix>
  struct MatrixTraits<EuclideanMean<RowCoefficients, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr std::size_t rows = row_dimension_of_v<NestedMatrix>;
    static constexpr std::size_t columns = column_dimension_of_v<NestedMatrix>;

  public:

    /// Make from a regular matrix. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
    (std::is_void_v<CC> or equivalent_to<CC, Axes<column_dimension_of_v<Arg>>>) and
    (row_dimension_of_v<Arg> == RC::euclidean_dimension)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      coefficients<RC> and (std::is_void_v<CC> or equivalent_to<CC, Axes<column_dimension_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_dimension_of<Arg>::value == RC::euclidean_dimension), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return EuclideanMean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Covariance<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<coefficients<C> and covariance_nestable<Arg>,int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<SquareRootCovariance<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this vector.
    static constexpr auto rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    /// Make SquareRootCovariance from a \ref covariance_nestable.
#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };

} // namespace OpenKalman

#endif //OPENKALMAN_MATRIXTRAITS_HPP
