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
    private:

      using Trait = EquivalentDenseWritableMatrix<std::decay_t<pattern_matrix_of_t<T>>, Scalar>;

    public:

      static constexpr bool is_writable = false;


      template<typename...D>
      static auto make_default(D&&...d)
      {
        return Trait::make_default(std::forward<D>(d)...);
      }

      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        return Trait::convert(std::forward<Arg>(arg));
      }

      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return Trait::to_native_matrix(std::forward<Arg>(arg));
      }
#ifdef __cpp_concepts
      template<index_descriptor...Ds, std::convertible_to<Scalar> ... Args>
#else
      template<typename...Ds, typename...Args, std::enable_if_t<(index_descriptor<Ds> and ...) and
        std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
      static auto make_from_elements(const std::tuple<Ds...>& d_tup, const Args ... args)
      {
        return Trait::make_from_elements(d_tup, args...);
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
      template<typename C, typename...D>
      static constexpr auto make_constant_matrix(C&& c, D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>>(std::forward<C>(c), std::forward<D>(d)...);
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
      template<typename C, typename...D>
      static constexpr auto make_constant_matrix(C&& c, D&&...d)
      {
        return make_constant_matrix_like<pattern_matrix_of_t<T>>(std::forward<C>(c), std::forward<D>(d)...);
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
    template<constant_matrix<CompileTimeStatus::any, Likelihood::maybe> NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix, CompileTimeStatus::any, Likelihood::maybe>>>
#endif
    {
      const DiagonalMatrix<NestedMatrix>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient {nested_matrix(xpr)};
      }
    };


#ifdef __cpp_concepts
    template<constant_diagonal_matrix<CompileTimeStatus::any, Likelihood::maybe> NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>>
#else
    template<typename NestedMatrix, TriangleType triangle_type>
    struct SingleConstant<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix, CompileTimeStatus::any, Likelihood::maybe>>>
#endif
    {
      const TriangularMatrix<NestedMatrix, triangle_type>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_diagonal_coefficient{nested_matrix(xpr)};
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix<CompileTimeStatus::any, Likelihood::maybe> NestedMatrix>
    struct SingleConstant<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
#else
    template<typename NestedMatrix>
    struct SingleConstant<TriangularMatrix<NestedMatrix, TriangleType::diagonal>, std::enable_if_t<
      constant_matrix<NestedMatrix, CompileTimeStatus::any, Likelihood::maybe>>>
#endif
    {
      const TriangularMatrix<NestedMatrix, TriangleType::diagonal>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient {nested_matrix(xpr)};
      }
    };


    template<typename NestedMatrix, HermitianAdapterType storage_type>
    struct SingleConstant<SelfAdjointMatrix<NestedMatrix, storage_type>>
    {
      const SelfAdjointMatrix<NestedMatrix, storage_type>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient{nested_matrix(xpr)};
      }

      constexpr auto get_constant_diagonal()
      {
        return constant_diagonal_coefficient {nested_matrix(xpr)};
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


    // ------------------ //
    //  TriangularTraits  //
    // ------------------ //

    template<typename NestedMatrix>
    struct TriangularTraits<DiagonalMatrix<NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      template<Likelihood b>
      static constexpr bool is_diagonal_adapter = true;
    };


    template<typename NestedMatrix, HermitianAdapterType storage_type>
    struct TriangularTraits<SelfAdjointMatrix<NestedMatrix, storage_type>>
    {
      template<TriangleType t, Likelihood>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, TriangleType::diagonal, Likelihood::maybe>;
    };


    template<typename NestedMatrix, TriangleType tt>
    struct TriangularTraits<TriangularMatrix<NestedMatrix, tt>>
    {
      template<TriangleType t, Likelihood>
      static constexpr bool is_triangular = t == TriangleType::any or tt == TriangleType::diagonal or tt == t or
        triangular_matrix<NestedMatrix, t, Likelihood::maybe>;

      static constexpr bool is_triangular_adapter = true;
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<ToEuclideanExpr<TypedIndex, NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = euclidean_index_descriptor<TypedIndex> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


    template<typename TypedIndex, typename NestedMatrix>
    struct TriangularTraits<FromEuclideanExpr<TypedIndex, NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = euclidean_index_descriptor<TypedIndex> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


    // ----------------- //
    //  HermitianTraits  //
    // ----------------- //

    template<typename NestedMatrix, HermitianAdapterType t>
    struct HermitianTraits<SelfAdjointMatrix<NestedMatrix, t>>
    {
      static constexpr bool is_hermitian = true;
      static constexpr HermitianAdapterType adapter_type = t;
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
    };


  } // namespace interface


  using namespace OpenKalman::internal;

  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

  template<typename N1, typename N2>
  struct is_modifiable_native<DiagonalMatrix<N1>, DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
  template<typename NestedMatrix, HermitianAdapterType storage_triangle, typename U> requires
    //(not hermitian_matrix<U>) or
    (eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, nested_matrix_of_t<U>>) or
    (not eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, storage_triangle>, U> : std::false_type {};
#else
  template<typename N1, HermitianAdapterType t1, typename N2, HermitianAdapterType t2>
  struct is_modifiable_native<SelfAdjointMatrix<N1, t1>, SelfAdjointMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, HermitianAdapterType t, typename U>
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*hermitian_matrix<U> and */modifiable<NestedMatrix, U>> {};
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
      not equivalent_to<C, row_index_descriptor_of_t<U>>)) or
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
      not equivalent_to<C, row_index_descriptor_of_t<U>>)) or
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
