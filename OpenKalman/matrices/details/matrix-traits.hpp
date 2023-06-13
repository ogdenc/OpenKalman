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

    // -------------------------------- //
    //   EquivalentDenseWritableMatrix  //
    // -------------------------------- //

#ifdef __cpp_concepts
    template<covariance T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<covariance<T>>>
#endif
    {
      static constexpr bool is_writable = EquivalentDenseWritableMatrix<std::decay_t<nested_matrix_of_t<T>>, Scalar>::is_writable;

      template<typename...D>
      static auto make_default(D&&...d)
      {
        return EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, Scalar>::make_default(std::forward<D>(d)...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, Scalar>;
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
    template<typed_matrix T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<typed_matrix<T>>>
#endif
    {
      static constexpr bool is_writable = EquivalentDenseWritableMatrix<std::decay_t<nested_matrix_of_t<T>>, Scalar>::is_writable;


      template<typename...D>
      static auto make_default(D&&...d)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, Scalar>;
        return Trait::make_default(std::forward<D>(d)...);
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using Trait = EquivalentDenseWritableMatrix<nested_matrix_of_t<T>, Scalar>;
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
        if constexpr (hermitian_matrix<NestedMatrix>)
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
        if constexpr (hermitian_matrix<NestedMatrix>)
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

#ifdef __cpp_concepts
    template<typed_matrix T>
    struct SingleConstant<T>
#else
    template<typename T>
    struct SingleConstant<T, std::enable_if_t<(typed_matrix<T>)>>>
#endif
      : SingleConstant<std::decay_t<nested_matrix_of_t<T>>> {};


#ifdef __cpp_concepts
    template<self_adjoint_covariance T> requires (not triangular_matrix<nested_matrix_of_t<T>, Likelihood::maybe>)
    struct SingleConstant<T>
#else
    template<typename T>
    struct SingleConstant<T, std::enable_if_t<(not triangular_matrix<nested_matrix_of_t<T>, Likelihood::maybe>)>>>
#endif
      : SingleConstant<std::decay_t<nested_matrix_of_t<T>>> {};


#ifdef __cpp_concepts
    template<triangular_covariance T> requires zero_matrix<nested_matrix_of_t<T>>
    struct SingleConstant<T>
#else
    template<typename T>
    struct SingleConstant<T, std::enable_if_t<triangular_covariance<T> and zero_matrix<nested_matrix_of_t<T>>>>>
#endif
      : SingleConstant<std::decay_t<nested_matrix_of_t<T>>> {};


      // ------------------------------------- //
      //   SingleConstantDiagonalMatrixTraits  //
      // ------------------------------------- //

    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        auto n = make_identity_matrix_like<NestedMatrix, Scalar>(std::forward<D>(d));
        return Matrix<RowCoeffs, ColCoeffs, std::decay_t<decltype(n)>>(std::move(n)) //< \ todo use make_matrix
      }
    };


    template<typename Coeffs, typename NestedMatrix, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Mean<Coeffs, NestedMatrix>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        auto n = make_identity_matrix_like<NestedMatrix, Scalar>(std::forward<D>(d));
        return Matrix<Coeffs, Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_matrix
      }
    };


    template<typename Coeffs, typename NestedMatrix, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<EuclideanMean<Coeffs, NestedMatrix>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        auto n = make_identity_matrix_like<NestedMatrix, Scalar>(std::forward<D>(d));
        return Matrix<Coeffs, Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_matrix
      }
    };


    template<typename Coeffs, typename NestedMatrix, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<Covariance<Coeffs, NestedMatrix>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        auto n = make_identity_matrix_like<NestedMatrix, Scalar>(std::forward<D>(d));
        return Covariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_covariance
      }
    };


    template<typename Coeffs, typename NestedMatrix, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<SquareRootCovariance<Coeffs, NestedMatrix>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        auto n = make_identity_matrix_like<NestedMatrix, Scalar>(std::forward<D>(d));
        return SquareRootCovariance<Coeffs, std::decay_t<decltype(n)>>(std::move(n)); //< \todo use make_square_root_covariance
      }
    };


    // ------------------ //
    //  TriangularTraits  //
    // ------------------ //

    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<SquareRootCovariance<TypedIndex, NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t, Likelihood::maybe> or
        hermitian_adapter<NestedMatrix, t == TriangleType::upper ? HermitianAdapterType::upper : HermitianAdapterType::lower>;

      static constexpr bool is_triangular_adapter = false;
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<Covariance<TypedIndex, NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, TriangleType::diagonal, Likelihood::maybe>;

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<typed_matrix T>
    struct TriangularTraits<T>
#else
    template<typename T>
    struct TriangularTraits<T, std::enable_if_t<typed_matrix<T>>>
#endif
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>> and
        triangular_matrix<nested_matrix_of_t<T>, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


    // ----------------- //
    //  HermitianTraits  //
    // ----------------- //

#ifdef __cpp_concepts
    template<self_adjoint_covariance T> requires hermitian_matrix<nested_matrix_of_t<T>>
    struct HermitianTraits<T>
#else
    template<typename T>
    struct HermitianTraits<T, std::enable_if_t<self_adjoint_covariance<T> and
      hermitian_matrix<nested_matrix_of<T>::type>>>
#endif
    {
      static constexpr bool is_hermitian = true;
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
    };


#ifdef __cpp_concepts
    template<typed_matrix T> requires hermitian_matrix<nested_matrix_of_t<T>> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>
    struct HermitianTraits<T>
#else
    template<typename T>
    struct HermitianTraits<T, std::enable_if_t<
      typed_matrix<T> and hermitian_matrix<nested_matrix_of<T>::type> and
      equivalent_to<row_coefficient_types_of_t<T>, column_coefficient_types_of_t<T>>>>
#endif
    {
      static constexpr bool is_hermitian = hermitian_matrix<nested_matrix_of_t<T>>;
    };

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
    static_assert(dimension_size_of_v<RowCoefficients> == rows);
    static_assert(dimension_size_of_v<ColCoefficients> == columns);

  public:

#ifdef __cpp_concepts
    template<fixed_index_descriptor RC = RowCoefficients, fixed_index_descriptor CC = ColumnCoefficients, typed_matrix_nestable Arg>
    requires (row_dimension_of_v<Arg> == dimension_size_of_v<RC>) and (column_dimension_of_v<Arg> == dimension_size_of_v<CC>)
#else
    template<typename RC = RowCoefficients, typename CC = ColumnCoefficients, typename Arg, std::enable_if_t<
      fixed_index_descriptor<RC> and fixed_index_descriptor<CC> and typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == dimension_size_of_v<RC>) and (column_dimension_of<Arg>::value == dimension_size_of_v<CC>), int> = 0>
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

    static_assert(dimension_size_of_v<RowCoefficients> == rows);

    /// Make from a typed_matrix_nestable. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<fixed_index_descriptor RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      (std::is_void_v<CC> or equivalent_to<CC, Dimensions<column_dimension_of_v<Arg>>>) and
      (row_dimension_of_v<Arg> == dimension_size_of_v<RC>)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      (std::is_void_v<CC> or equivalent_to<CC, Dimensions<column_dimension_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_dimension_of<Arg>::value == dimension_size_of_v<RC>), int> = 0>
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
    template<fixed_index_descriptor RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
    (std::is_void_v<CC> or equivalent_to<CC, Dimensions<column_dimension_of_v<Arg>>>) and
    (row_dimension_of_v<Arg> == euclidean_dimension_size_of_v<RC>)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      fixed_index_descriptor<RC> and (std::is_void_v<CC> or equivalent_to<CC, Dimensions<column_dimension_of<Arg>::value>>) and
      typed_matrix_nestable<Arg> and (row_dimension_of<Arg>::value == euclidean_dimension_size_of_v<RC>), int> = 0>
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
    template<fixed_index_descriptor C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<fixed_index_descriptor<C> and covariance_nestable<Arg>,int> = 0>
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
    template<fixed_index_descriptor C = Coeffs, covariance_nestable Arg>
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
