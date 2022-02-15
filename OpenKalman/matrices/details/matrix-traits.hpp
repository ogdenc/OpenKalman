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

    // ------------- //
    //  RowExtentOf  //
    // ------------- //

#ifdef __cpp_concepts
    template<typed_matrix T>
    struct RowExtentOf<T>
#else
    template<typename T>
    struct RowExtentOf<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : RowExtentOf<nested_matrix_of_t<T>>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
        if constexpr (dynamic_rows<Arg>)
          return row_count(nested_matrix(std::forward<Arg>(arg)));
        else
          return RowExtentOf::value;
      }
    };


#ifdef __cpp_concepts
    template<covariance T>
    struct RowExtentOf<T>
#else
    template<typename T>
    struct RowExtentOf<T, std::enable_if_t<covariance<T>>>
#endif
      : std::integral_constant<std::size_t, dynamic_rows<nested_matrix_of_t<T>> ?
        column_extent_of_v<nested_matrix_of_t<T>> : row_extent_of_v<nested_matrix_of_t<T>>>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
        using N = nested_matrix_of_t<T>;

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


    // ---------------- //
    //  ColumnExtentOf  //
    // ---------------- //

#ifdef __cpp_concepts
    template<typed_matrix T>
    struct ColumnExtentOf<T>
#else
    template<typename T>
    struct ColumnExtentOf<T, std::enable_if_t<typed_matrix<T>>>
#endif
      : ColumnExtentOf<nested_matrix_of_t<T>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
        if constexpr (dynamic_columns<Arg>)
          return column_count(nested_matrix(std::forward<Arg>(arg)));
        else
          return ColumnExtentOf::value;
      }
    };


#ifdef __cpp_concepts
    template<covariance T>
    struct ColumnExtentOf<T>
#else
    template<typename T>
    struct ColumnExtentOf<T, std::enable_if_t<covariance<T>>>
#endif
      : std::integral_constant<std::size_t, dynamic_columns<nested_matrix_of_t<T>> ?
        row_extent_of_v<nested_matrix_of_t<T>> : column_extent_of_v<nested_matrix_of_t<T>>>
    {
      template<typename Arg>
      static constexpr std::size_t columns_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
        using N = nested_matrix_of_t<T>;

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


    // -------------- //
    //  ScalarTypeOf  //
    // -------------- //

#ifdef __cpp_concepts
    template<typename T> requires typed_matrix<T> or covariance<T>
    struct ScalarTypeOf<T>
#else
    template<typename T>
    struct ScalarTypeOf<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
#endif
      : ScalarTypeOf<nested_matrix_of_t<T>> {};


    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<covariance T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<covariance<T>>>
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
        return Base::convert(OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg)));
      }
    };


#ifdef __cpp_concepts
    template<typed_matrix T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<typed_matrix<T>>>
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
        return Base::convert(nested_matrix(std::forward<Arg>(arg)));
      }
    };


    // ----------------- //
    //   Dependencies  //
    // ----------------- //

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


  } // namespace interface


  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix>
  struct MatrixTraits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr std::size_t rows = row_extent_of_v<NestedMatrix>;
    static constexpr std::size_t columns = column_extent_of_v<NestedMatrix>;
    static_assert(RowCoeffs::dimensions == rows);
    static_assert(ColumnCoeffs::dimensions == columns);

  public:

    using Coefficients = RowCoeffs;
    using RowCoefficients = RowCoeffs;
    using ColumnCoefficients = ColCoeffs;


#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, coefficients CC = ColumnCoefficients, typed_matrix_nestable Arg>
    requires (row_extent_of_v<Arg> == RC::dimensions) and (column_extent_of_v<Arg> == CC::dimensions)
#else
    template<typename RC = RowCoefficients, typename CC = ColumnCoefficients, typename Arg, std::enable_if_t<
      coefficients<RC> and coefficients<CC> and typed_matrix_nestable<Arg> and
      (row_extent_of<Arg>::value == RC::dimensions) and (column_extent_of<Arg>::value == CC::dimensions), int> = 0>
#endif
    static auto make(Arg&& arg)
    {
      return Matrix<RC, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      auto b = MatrixTraits<NestedMatrix>::identity(args...);
      return make<RowCoefficients, RowCoefficients>(std::move(b));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Mean<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = column_extent_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(RowCoefficients::dimensions == rows);

    /// Make from a typed_matrix_nestable. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      (std::is_void_v<CC> or equivalent_to<CC, Axes<column_extent_of_v<Arg>>>) and
      (row_extent_of_v<Arg> == RC::dimensions)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      (std::is_void_v<CC> or equivalent_to<CC, Axes<column_extent_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_extent_of<Arg>::value == RC::dimensions), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg)); using B = decltype(b);
      return Mean<RC, std::decay_t<B>>(std::forward<B>(b));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      auto b = MatrixTraits<NestedMatrix>::identity(args...);
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<EuclideanMean<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr std::size_t rows = row_extent_of_v<NestedMatrix>;
    static constexpr std::size_t columns = column_extent_of_v<NestedMatrix>;

  public:

    using RowCoefficients = Coeffs;
    static_assert(RowCoefficients::euclidean_dimensions == rows);

    using ColumnCoefficients = Axes<columns>;


    /// Make from a regular matrix. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
    (std::is_void_v<CC> or equivalent_to<CC, Axes<column_extent_of_v<Arg>>>) and
    (row_extent_of_v<Arg> == RC::euclidean_dimensions)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      coefficients<RC> and (std::is_void_v<CC> or equivalent_to<CC, Axes<column_extent_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_extent_of<Arg>::value == RC::euclidean_dimensions), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return EuclideanMean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      auto b = MatrixTraits<NestedMatrix>::identity(args...);
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<Covariance<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>;
    static constexpr auto rows = row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Coeffs;


#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<coefficients<C> and covariance_nestable<Arg>,int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 2 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 2 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::identity(args...));
    }

  };


  template<typename Coeffs, typename NestedMatrix>
  struct MatrixTraits<SquareRootCovariance<Coeffs, NestedMatrix>>
  {
  private:

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this vector.
    static constexpr auto rows = row_extent_of_v<NestedMatrix>;
    static constexpr auto columns = rows;

  public:

    static_assert(Coeffs::dimensions == rows);
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Coeffs;

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


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 2 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 2 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (dynamic_coefficients<Coeffs> ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::identity(args...));
    }

  };

}

#endif //OPENKALMAN_MATRIXTRAITS_HPP
