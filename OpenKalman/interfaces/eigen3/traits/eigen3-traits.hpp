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
 * \brief Type traits as applied to native Eigen3 types.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_HPP
#define OPENKALMAN_EIGEN3_TRAITS_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace EGI = Eigen::internal;


#ifdef __cpp_concepts
  template<native_eigen_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<native_eigen_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, native_eigen_matrix M>
#else
  template<TriangleType t = TriangleType::lower, typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<native_eigen_matrix M>
#else
  template<typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, native_eigen_matrix M>
#else
 template<TriangleType t = TriangleType::lower, typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


  namespace interface
  {
    using namespace Eigen3;

  // ---------------------- //
  //  native_eigen_general  //
  // ---------------------- //

#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexibleObjectTraits<T>
#else
    template<typename T>
    struct IndexibleObjectTraits<T, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      static constexpr std::size_t max_indices = 2;
      using scalar_type = typename std::decay_t<T>::Scalar;
    };


#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexTraits<T, 0>
#else
    template<typename T>
    struct IndexTraits<T, 0, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      static constexpr auto e_dim =
        (std::decay_t<T>::RowsAtCompileTime == Eigen::Dynamic) and
          (eigen_SelfAdjointView<T> or eigen_TriangularView<T>) ? std::decay_t<T>::ColsAtCompileTime :
        std::decay_t<T>::RowsAtCompileTime;

    public:

      static constexpr std::size_t dimension = e_dim == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(e_dim);

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_rows<Arg>)
          return static_cast<std::size_t>(arg.rows());
        else
          return dimension;
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexTraits<T, 1>
#else
    template<typename T>
    struct IndexTraits<T, 1, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      static constexpr auto e_dim =
        (std::decay_t<T>::ColsAtCompileTime == Eigen::Dynamic) and (
          eigen_SelfAdjointView<T> or eigen_TriangularView<T>) ? std::decay_t<T>::RowsAtCompileTime :
        std::decay_t<T>::ColsAtCompileTime;

    public:

      static constexpr std::size_t dimension = e_dim == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(e_dim);

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_columns<Arg>)
          return static_cast<std::size_t>(arg.cols());
        else
          return dimension;
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      template<Eigen::Index...Args>
      using dense_type = std::conditional_t<native_eigen_array<T>,
        Eigen::Array<Scalar, Args...>, Eigen::Matrix<Scalar, Args...>>;

      template<std::size_t...Args>
      using type = dense_type<(Args == dynamic_size ? Eigen::Dynamic : static_cast<Eigen::Index>(Args))...>;

#  ifdef __cpp_concepts
      template<typename ArgType>
#  else
      template<typename ArgType, typename = void>
#  endif
      struct traits_and_evaluator_defined : std::false_type {};


#  ifdef __cpp_concepts
       template<typename ArgType> requires
         requires { typename Eigen::internal::traits<ArgType>; typename Eigen::internal::evaluator<ArgType>; }
       struct traits_and_evaluator_defined<ArgType> : std::true_type {};
#  else
       template<typename ArgType>
       struct traits_and_evaluator_defined<ArgType, std::enable_if_t<
         std::is_void<std::void_t<Eigen::internal::traits<ArgType>>>::value and
         std::is_void<std::void_t<Eigen::internal::evaluator<ArgType>>>::value>> : std::true_type {};
#  endif

    public:

      static constexpr bool is_writable = (native_eigen_matrix<T> or native_eigen_array<T>) and
        static_cast<bool>(Eigen::internal::traits<std::decay_t<T>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit));


      template<typename...D>
      static auto make_default(D&&...d)
      {
        using M = type<dimension_size_of_v<D>...>;

        if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
          return M(static_cast<Eigen::Index>(get_dimension_size_of(d))...);
        else
          return M {};
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using M = type<index_dimension_of_v<Arg, 0>, index_dimension_of_v<Arg, 1>>;

        if constexpr (eigen_DiagonalWrapper<Arg>)
        {
          // Note: Arg's nested matrix might not be a column vector.
          return M {to_diagonal(Conversions<Arg>::diagonal_of(std::forward<Arg>(arg)))};
        }
        else if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
          not std::is_const_v<std::remove_reference_t<Arg>>)
        {
          return std::forward<Arg>(arg);
        }
        else if constexpr (std::is_constructible_v<M, Arg&&>)
        {
          return M {std::forward<Arg>(arg)};
        }
        else
        {
          auto r = get_index_dimension_of<0>(arg);
          auto c = get_index_dimension_of<1>(arg);
          auto m = make_default(r, c);
          for (int i = 0; i < r ; ++i) for (int j = 0; j < c ; ++j)
          {
            set_element(m, get_element(std::forward<Arg>(arg), i, j), i, j);
          }
          return m;
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        if constexpr (native_eigen_matrix<Arg>)
          return std::forward<Arg>(arg);
        else if constexpr (native_eigen_array<Arg>)
          return std::forward<Arg>(arg).matrix();
        else if constexpr (native_eigen_general<Arg>)
          return convert(std::forward<Arg>(arg));
        else
        {
          static_assert(traits_and_evaluator_defined<Arg>::value, "To convert to a native Eigen matrix, the interface "
            "must define a trait and an evaluator for the argument");
          return EigenWrapper<std::decay_t<Arg>> {std::forward<Arg>(arg)};
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(const EigenWrapper<Arg>& arg) { return arg; }


      template<typename Arg>
      static decltype(auto) to_native_matrix(EigenWrapper<Arg>&& arg) { return std::move(arg); }

    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      template<typename...Ds>
      static constexpr auto make_zero_matrix(Ds&&...ds)
      {
        using N = dense_writable_matrix_t<T, Scalar, std::decay_t<Ds>...>;
        return ZeroAdapter<N> {std::forward<Ds>(ds)...};
      }

      template<auto...constant, typename...Ds>
      static constexpr auto make_constant_matrix(Ds&&...ds)
      {
        using N = dense_writable_matrix_t<T, Scalar, std::decay_t<Ds>...>;
        return ConstantAdapter<N, constant...> {std::forward<Ds>(ds)...};
      }

      template<typename S, typename...Ds>
      static constexpr auto make_runtime_constant(S&& s, Ds&&...ds)
      {
        static_assert(sizeof...(Ds) == 2);
        using N = dense_writable_matrix_t<T, S, std::decay_t<Ds>...>;
        return N::Constant(static_cast<Eigen::Index>(get_dimension_size_of(std::forward<Ds>(ds)))..., std::forward<S>(s));
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        if constexpr (dimension_size_of_v<D> == dynamic_size)
        {
          return to_diagonal(make_constant_matrix_like<T, Scalar, 1>(std::forward<D>(d), Dimensions<1>{}));
        }
        else
        {
          constexpr Eigen::Index n {dimension_size_of_v<D>};
          return eigen_matrix_t<Scalar, n, n>::Identity();
        }
      }
    };


    // ------- //
    //  Array  //
    // ------- //

    template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    struct Dependencies<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };


    // -------------- //
    //  ArrayWrapper  //
    // -------------- //

    template<typename XprType>
    struct Dependencies<Eigen::ArrayWrapper<XprType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename Eigen::ArrayWrapper<XprType>::NestedExpressionType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::ArrayWrapper<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename XprType>
    struct SingleConstant<Eigen::ArrayWrapper<XprType>> : SingleConstant<std::decay_t<XprType>>
    {
      SingleConstant(const Eigen::ArrayWrapper<XprType>& xpr) :
        SingleConstant<std::decay_t<XprType>> {xpr.nestedExpression()} {};
    };


    template<typename XprType>
    struct DiagonalTraits<Eigen::ArrayWrapper<XprType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<XprType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix XprType>
    struct TriangularTraits<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct TriangularTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<triangular_matrix<XprType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // ------- //
    //  Block  //
    // ------- //

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct Dependencies<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename EGI::ref_selector<XprType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::Block should always be converted to Matrix

    };


    // A block taken from a constant matrix is constant.
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct SingleConstant<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
      const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr.nestedExpression()};
      }
    };


    // ---------- //
    //  Diagonal  //
    // ---------- //

    template<typename MatrixType, int DiagIndex>
    struct Dependencies<Eigen::Diagonal<MatrixType, DiagIndex>>
    {
      static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;

      using type = std::tuple<typename EGI::ref_selector<MatrixType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Rely on default for convert_to_self_contained. Should always convert to a dense, writable matrix.

    };


    template<typename MatrixType, int DiagIndex>
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>>
    {
      const Eigen::Diagonal<MatrixType, DiagIndex>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (constant_diagonal_matrix<MatrixType, Likelihood::maybe>)
        {
          if constexpr (DiagIndex == Eigen::DynamicIndex)
          {
            if (xpr.index() == 0)
              return constant_diagonal_coefficient{xpr.nestedExpression()}();
            else
              return scalar_type_of_t<MatrixType>{0};
          }
          else
          {
            if constexpr (DiagIndex == 0)
              return constant_diagonal_coefficient{xpr.nestedExpression()};
            else
              return std::integral_constant<int, 0>{};
          }
        }
        else
        {
          return constant_coefficient{xpr.nestedExpression()};
        }
      }
    };


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct Dependencies<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<
        typename Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>::DiagonalVectorType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).diagonal();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct DiagonalTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct TriangularTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr TriangleType triangle_type = TriangleType::diagonal;
      static constexpr bool is_triangular_adapter = false;
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalMatrix.
   */
  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct MatrixTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : MatrixTraits<Eigen::Matrix<Scalar, SizeAtCompileTime, SizeAtCompileTime>>
  {
  };


  // ----------------- //
  //  DiagonalWrapper  //
  // ----------------- //

  namespace interface
  {

    template<typename DiagVectorType>
    struct Dependencies<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename DiagVectorType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).diagonal();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename DiagonalVectorType>
    struct SingleConstant<Eigen::DiagonalWrapper<DiagonalVectorType>>
    {
      const Eigen::DiagonalWrapper<DiagonalVectorType>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient {xpr.diagonal()};
      }
    };


    template<typename DiagonalVectorType>
    struct DiagonalTraits<Eigen::DiagonalWrapper<DiagonalVectorType>>
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename DiagonalVectorType>
    struct TriangularTraits<Eigen::DiagonalWrapper<DiagonalVectorType>>
    {
      static constexpr TriangleType triangle_type = TriangleType::diagonal;
      static constexpr bool is_triangular_adapter = false;
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalWrapper.
   */
  template<typename V>
  struct MatrixTraits<Eigen::DiagonalWrapper<V>>
    : MatrixTraits<Eigen::Matrix<typename EGI::traits<std::decay_t<V>>::Scalar,
        V::SizeAtCompileTime, V::SizeAtCompileTime>>
  {
  };


  // ------------------------- //
  //  IndexedView (Eigen 3.4)  //
  // ------------------------- //


  // ------------- //
  //  Homogeneous  //
  // ------------- //
  // \todo: Add. This is a child of Eigen::MatrixBase


  namespace interface
  {

    // --------- //
    //  Inverse  //
    // --------- //

    template<typename XprType>
    struct Dependencies<Eigen::Inverse<XprType>>
    {
    private:

      using T = Eigen::Inverse<XprType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::XprTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::Inverse<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // ----- //
    //  Map  //
    // ----- //

    template<typename PlainObjectType, int MapOptions, typename StrideType>
    struct Dependencies<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    {
    private:
      using M = Eigen::Map<PlainObjectType, MapOptions, StrideType>;
    public:
      static constexpr bool has_runtime_parameters =
        M::RowsAtCompileTime == Eigen::Dynamic or M::ColsAtCompileTime == Eigen::Dynamic or
        M::OuterStrideAtCompileTime == Eigen::Dynamic or M::InnerStrideAtCompileTime == Eigen::Dynamic;

      // Map is not self-contained in any circumstances.
      using type = std::tuple<decltype(*std::declval<typename M::PointerType>())>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return *std::forward<Arg>(arg).data();
      }
    };


    // -------- //
    //  Matrix  //
    // -------- //

    template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
    struct Dependencies<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };


    // --------------- //
    //  MatrixWrapper  //
    // --------------- //

    template<typename XprType>
    struct Dependencies<Eigen::MatrixWrapper<XprType>>
    {
    private:

      using T = Eigen::MatrixWrapper<XprType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::NestedExpressionType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::MatrixWrapper<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename XprType>
    struct SingleConstant<Eigen::MatrixWrapper<XprType>> : SingleConstant<std::decay_t<XprType>>
    {
      SingleConstant(const Eigen::MatrixWrapper<XprType>& xpr) :
        SingleConstant<std::decay_t<XprType>> {xpr.nestedExpression()} {};
    };


    template<typename XprType>
    struct DiagonalTraits<Eigen::MatrixWrapper<XprType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<XprType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix XprType>
    struct TriangularTraits<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct TriangularTraits<Eigen::MatrixWrapper<XprType>, std::enable_if_t<triangular_matrix<XprType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ------------------ //
  //  PartialReduxExpr  //
  // ------------------ //

    template<typename MatrixType, typename MemberOp, int Direction>
    struct Dependencies<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename MatrixType::Nested, const MemberOp>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        if constexpr (i == 0)
          return std::forward<Arg>(arg).nestedExpression();
        else
          return std::forward<Arg>(arg).functor();
        static_assert(i <= 1);
      }

      // If a partial redux expression needs to be partially evaluated, it's probably faster to do a full evaluation.
      // Thus, we omit the conversion function.
    };


    template<typename MatrixType, typename MemberOp, int Direction>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    {
    private:

      using Scalar = scalar_type_of_t<MatrixType>;

      template<bool is_diag, typename> struct Op;

      template<bool is_diag, int p, typename...Args>
      struct Op<is_diag, EGI::member_lpnorm<p, Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = x >= 0 ? x : -x;
          if constexpr (is_diag)
          {
            if constexpr (p == 0) return dim;
            else return arg;
          }
          else
          {
            if constexpr (p == 2) return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
            else if constexpr (p == 1) return dim * arg;
            else if constexpr (p == 0) return dim;
            else if constexpr (p == Eigen::Infinity) return arg;
            else return internal::constexpr_pow(static_cast<Scalar>(dim), 1.0/p) * arg;
          }
        }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_norm<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = x >= 0 ? x : -x;
          if constexpr (is_diag) return arg;
          else return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_stableNorm<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = x >= 0 ? x : -x;
          if constexpr (is_diag) return arg;
          else return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_hypotNorm<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = x >= 0 ? x : -x;
          if constexpr (is_diag) return arg;
          else return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      };

# if not EIGEN_VERSION_AT_LEAST(3,4,0)
      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_squaredNorm<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return (is_diag ? 1 : dim) * x * x; }
      };

      template<bool is_diag, typename...Args>
      struct V<is_diag, EGI::member_mean<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          if constexpr (is_diag) return x / dim;
          else return x;
        }
      };
# endif

      // \todo add EGI::member_redux<Args...>

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_sum<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return (is_diag ? 1 : dim) * x; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_minCoeff<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return x; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_maxCoeff<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return x; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_all<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return x; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_any<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return x; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_count<Args...>>
      {
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept { return dim; }
      };

      template<bool is_diag, typename...Args>
      struct Op<is_diag, EGI::member_prod<Args...>>
      {
        constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
        { 
          if (dim == 0) return 0;
          else if (dim == 1) return x;
          else if constexpr (is_diag) return 0;
          else return OpenKalman::internal::constexpr_pow(x, dim);
        }
      };

    public:

      const Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>& xpr;

      constexpr auto get_constant()
      {
        //constexpr auto dim = Direction == Eigen::Horizontal ? index_dimension_of_v<MatrixType, 1> : index_dimension_of_v<MatrixType, 0>;

        auto dim = [](const auto& x){
          constexpr auto i = Direction == Eigen::Horizontal ? 1 : 0;
          constexpr auto d = index_dimension_of_v<MatrixType, i>;
          if constexpr (d == dynamic_size) return get_index_dimension_of<i>(x.nestedExpression());
          else return std::integral_constant<std::size_t, d>{};
        }(xpr);

        if constexpr (constant_diagonal_matrix<MatrixType, Likelihood::maybe>)
        {
          return scalar_constant_operation {Op<true, MemberOp>{}, constant_diagonal_coefficient{xpr.nestedExpression()}, dim};
        }
        else
        {
          return scalar_constant_operation {Op<false, MemberOp>{}, constant_coefficient{xpr.nestedExpression()}, dim};
        }
      }
    };


    // ------------------- //
    //  PermutationMatrix  //
    // ------------------- //

    template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
    struct Dependencies<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<
        typename Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>::IndicesType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).indices();
      }

      // PermutationMatrix is always self-contained.

    };


    // -------------------- //
    //  PermutationWrapper  //
    // -------------------- //

    template<typename IndicesType>
    struct Dependencies<Eigen::PermutationWrapper<IndicesType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename IndicesType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).indices();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using NewIndicesType = equivalent_self_contained_t<IndicesType>;
        if constexpr (not std::is_lvalue_reference_v<typename NewIndicesType::Nested>)
          return Eigen::PermutationWrapper<NewIndicesType> {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // --------- //
    //  Product  //
    // --------- //

    template<typename LhsType, typename RhsType, int Option>
    struct Dependencies<Eigen::Product<LhsType, RhsType, Option>>
    {
    private:

      using T = Eigen::Product<LhsType, RhsType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::LhsNested, typename T::RhsNested >;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).lhs();
        else
          return std::forward<Arg>(arg).rhs();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::Product<equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>, Option>;
        constexpr Eigen::Index to_be_evaluated_size = self_contained<LhsType> ?
          RhsType::RowsAtCompileTime * RhsType::ColsAtCompileTime :
          LhsType::RowsAtCompileTime * LhsType::ColsAtCompileTime;

        // Do a partial evaluation if at least one argument is self-contained and result size > non-self-contained size.
        if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
          (LhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (LhsType::ColsAtCompileTime != Eigen::Dynamic) and
          (RhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (RhsType::ColsAtCompileTime != Eigen::Dynamic) and
          ((Eigen::Index)LhsType::RowsAtCompileTime * (Eigen::Index)RhsType::ColsAtCompileTime > to_be_evaluated_size) and
          not std::is_lvalue_reference_v<typename N::LhsNested> and
          not std::is_lvalue_reference_v<typename N::RhsNested>)
        {
          return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs())};
        }
        else
        {
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
        }
      }
    };


    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>>
    {
      const Eigen::Product<Arg1, Arg2>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (constant_diagonal_matrix<Arg1, Likelihood::maybe> and constant_matrix<Arg2, Likelihood::maybe>)
        {
          return scalar_constant_operation {std::multiplies<>{},
            constant_diagonal_coefficient{xpr.lhs()}, constant_coefficient{xpr.rhs()}};
        }
        else if constexpr (constant_matrix<Arg1, Likelihood::maybe> and constant_diagonal_matrix<Arg2, Likelihood::maybe>)
        {
          return scalar_constant_operation {std::multiplies<>{},
            constant_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
        }
        else
        {
          struct Op
          {
            constexpr auto operator()(std::size_t dim, scalar_type_of_t<Arg1> arg1, scalar_type_of_t<Arg2> arg2) const noexcept
            {
              return dim * arg1 * arg2;
            }
          };

          constexpr auto dim = dynamic_dimension<Arg1, 1> ? index_dimension_of_v<Arg2, 0> : index_dimension_of_v<Arg1, 1>;
          if constexpr (dim == dynamic_size)
          {
            return scalar_constant_operation {Op{}, get_index_dimension_of<1>(xpr.lhs()),
              constant_coefficient{xpr.rhs()}, constant_coefficient{xpr.lhs()}};
          }
          else
          {
            if constexpr (zero_matrix<Arg1>) return constant_coefficient{xpr.lhs()};
            else if constexpr (zero_matrix<Arg2>) return constant_coefficient{xpr.rhs()};
            else if constexpr (constant_diagonal_matrix<Arg1, Likelihood::maybe>) return scalar_constant_operation {
              std::multiplies<>{}, constant_diagonal_coefficient{xpr.lhs()}, constant_coefficient{xpr.rhs()}};
            else if constexpr (constant_diagonal_matrix<Arg2, Likelihood::maybe>) return scalar_constant_operation {
              std::multiplies<>{}, constant_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
            else return scalar_constant_operation {Op{}, std::integral_constant<std::size_t, dim>{},
              constant_coefficient{xpr.rhs()}, constant_coefficient{xpr.lhs()}};
          }
        }
      }
      
      constexpr auto get_constant_diagonal()
      {
        return scalar_constant_operation {std::multiplies<>{},
          constant_diagonal_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
      }
    };


    /// The product of two diagonal matrices is also diagonal.
    template<typename Arg1, typename Arg2>
    struct DiagonalTraits<Eigen::Product<Arg1, Arg2>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg1> and diagonal_matrix<Arg2>;
    };


    /// A diagonal matrix times a triangular matrix is triangular.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, triangular_matrix Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      diagonal_matrix<Arg1> and triangular_matrix<Arg2>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg2>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// A triangular matrix times a diagonal matrix is triangular.
#ifdef __cpp_concepts
    template<triangular_matrix Arg1, diagonal_matrix Arg2> requires (not diagonal_matrix<Arg1>)
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      triangular_matrix<Arg1> and diagonal_matrix<Arg2> and not diagonal_matrix<Arg1>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg1>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// A constant diagonal matrix times a self-adjoint matrix (or vice versa) is self-adjoint.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix Arg1, hermitian_matrix Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (constant_diagonal_matrix<Arg1> and hermitian_matrix<Arg2>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    /// A self-adjoint matrix times a constant-diagonal matrix is self-adjoint.
#ifdef __cpp_concepts
    template<hermitian_matrix Arg1, constant_diagonal_matrix Arg2> requires (not constant_diagonal_matrix<Arg1>)
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (hermitian_matrix<Arg1> and constant_diagonal_matrix<Arg2> and not constant_diagonal_matrix<Arg1>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ----- //
  //  Ref  //
  // ----- //

    template<typename PlainObjectType, int Options, typename StrideType>
    struct Dependencies<Eigen::Ref<PlainObjectType, Options, StrideType>>
    {
      static constexpr bool has_runtime_parameters = false;
      // Ref is not self-contained in any circumstances.
    };


  // ----------- //
  //  Replicate  //
  // ----------- //

    template<typename MatrixType, int RowFactor, int ColFactor>
    struct Dependencies<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    {
    private:

      using T = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;

    public:

      static constexpr bool has_runtime_parameters = RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic;
      using type = std::tuple<typename EGI::traits<T>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::Replicate<equivalent_self_contained_t<MatrixType>, RowFactor, ColFactor>;
        if constexpr (not std::is_lvalue_reference_v<typename EGI::traits<N>::MatrixTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename MatrixType, int RowFactor, int ColFactor>
    struct SingleConstant<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    {
      const Eigen::Replicate<MatrixType, RowFactor, ColFactor>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr.nestedExpression()};
      }
    };


    template<typename MatrixType>
    struct SingleConstant<Eigen::Replicate<MatrixType, 1, 1>> : SingleConstant<std::decay_t<MatrixType>>
    {
      SingleConstant(const Eigen::Replicate<MatrixType, 1, 1>& xpr) :
        SingleConstant<std::decay_t<MatrixType>> {xpr.nestedExpression()} {};
    };


    template<typename MatrixType>
    struct DiagonalTraits<Eigen::Replicate<MatrixType, 1, 1>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType>
    struct TriangularTraits<Eigen::Replicate<MatrixType, 1, 1>>
#else
    template<typename MatrixType>
    struct TriangularTraits<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<triangular_matrix<MatrixType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<MatrixType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<hermitian_matrix<MatrixType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // --------- //
  //  Reverse  //
  // --------- //

    template<typename MatrixType, int Direction>
    struct Dependencies<Eigen::Reverse<MatrixType, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename MatrixType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using M = equivalent_self_contained_t<MatrixType>;
        if constexpr (not std::is_lvalue_reference_v<typename M::Nested>)
          return Eigen::Reverse<M, Direction> {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename MatrixType, int Direction>
    struct SingleConstant<Eigen::Reverse<MatrixType, Direction>>
    {
      const Eigen::Reverse<MatrixType, Direction>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr.nestedExpression()};
      }
    };


    template<typename MatrixType>
    struct SingleConstant<Eigen::Reverse<MatrixType, Eigen::BothDirections>> : SingleConstant<std::decay_t<MatrixType>>
    {
      SingleConstant(const Eigen::Reverse<MatrixType>& xpr) :
        SingleConstant<std::decay_t<MatrixType>> {xpr.nestedExpression()} {};
    };


    template<typename MatrixType, int Direction>
    struct DiagonalTraits<Eigen::Reverse<MatrixType, Direction>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType> and
        (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>);
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType, int Direction> requires
      (Direction == Eigen::BothDirections) or (one_by_one_matrix<MatrixType>)
    struct TriangularTraits<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct TriangularTraits<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<triangular_matrix<MatrixType> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)>>
#endif
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (lower_triangular_matrix<MatrixType> ? TriangleType::upper : TriangleType::lower);
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType, int Direction> requires
      (Direction == Eigen::BothDirections) or (one_by_one_matrix<MatrixType>)
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<hermitian_matrix<MatrixType> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // -------- //
    //  Select  //
    // -------- //

    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct Dependencies<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    {
    private:

      using T = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename ConditionMatrixType::Nested, typename ThenMatrixType::Nested,
        typename ElseMatrixType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 3);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).conditionMatrix();
        else if constexpr (i == 1)
          return std::forward<Arg>(arg).thenMatrix();
        else
          return std::forward<Arg>(arg).elseMatrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::Select<equivalent_self_contained_t<ConditionMatrixType>,
          equivalent_self_contained_t<ThenMatrixType>, equivalent_self_contained_t<ElseMatrixType>>;
        // Do a partial evaluation as long as at least two arguments are already self-contained.
        if constexpr (
          ((self_contained<ConditionMatrixType> ? 1 : 0) + (self_contained<ThenMatrixType> ? 1 : 0) +
            (self_contained<ElseMatrixType> ? 1 : 0) >= 2) and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ConditionMatrixType>::Nested> and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ThenMatrixType>::Nested> and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ElseMatrixType>::Nested>)
        {
          return N {make_self_contained(arg.arg1()), make_self_contained(arg.arg2()), make_self_contained(arg.arg3()),
                    arg.functor()};
        }
        else
        {
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
        }
      }
    };


    // --- constant_coefficient --- //

    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    {
      const Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (constant_matrix<ConditionMatrixType, Likelihood::maybe>)
        {
          if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
            return constant_coefficient{xpr.thenMatrix()};
          else
            return constant_coefficient{xpr.elseMatrix()};
        }
        else if constexpr (constant_matrix<ThenMatrixType, Likelihood::maybe> and constant_matrix<ElseMatrixType, Likelihood::maybe>)
        {
          if constexpr (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
            return constant_coefficient{xpr.thenMatrix()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr (constant_matrix<ConditionMatrixType, Likelihood::maybe>)
        {
          if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
            return constant_diagonal_coefficient{xpr.thenMatrix()};
          else
            return constant_diagonal_coefficient{xpr.elseMatrix()};
        }
        else if constexpr (constant_diagonal_matrix<ThenMatrixType, Likelihood::maybe> and constant_diagonal_matrix<ElseMatrixType, Likelihood::maybe>)
        {
          if constexpr (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
            return constant_diagonal_coefficient{xpr.thenMatrix()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct DiagonalTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct DiagonalTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    {
      static constexpr bool is_diagonal =
        (diagonal_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (diagonal_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>);
    };


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
      (triangular_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
      (triangular_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
      constant_matrix<ConditionMatrixType> and
      ((triangular_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
       (triangular_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
    {
      static constexpr TriangleType triangle_type = static_cast<bool>(constant_coefficient_v<ConditionMatrixType>) ?
        triangle_type_of_v<ThenMatrixType> : triangle_type_of_v<ElseMatrixType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
      (hermitian_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
      (hermitian_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
      constant_matrix<ConditionMatrixType> and
      ((hermitian_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
       (hermitian_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix ConditionMatrixType, hermitian_matrix ThenMatrixType,
      hermitian_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>)
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<hermitian_matrix<ConditionMatrixType> and hermitian_matrix<ThenMatrixType> and
        hermitian_matrix<ElseMatrixType> and (not constant_matrix<ConditionMatrixType>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ----------------- //
  //  SelfAdjointView  //
  // ----------------- //

    template<typename M, unsigned int UpLo>
    struct Dependencies<Eigen::SelfAdjointView<M, UpLo>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename Eigen::SelfAdjointView<M, UpLo>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(SelfAdjointMatrix {std::forward<Arg>(arg)});
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstant<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
    private:

      struct C
      {
        using value_type = scalar_type_of_t<MatrixType>;
        static constexpr value_type value = 1;
        static constexpr Likelihood status = Likelihood::definitely;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
      };

    public:

      const Eigen::SelfAdjointView<MatrixType, UpLo>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (not complex_number<scalar_type_of_t<MatrixType>>)
          return constant_coefficient{xpr.nestedExpression()};
        else if constexpr (constant_matrix<MatrixType, Likelihood::maybe, CompileTimeStatus::known>)
        {
          if constexpr (real_axis_number<constant_coefficient<MatrixType>>)
            return constant_coefficient{xpr.nestedExpression()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr (eigen_Identity<MatrixType>) return C{};
        else return constant_diagonal_coefficient {xpr.nestedExpression()};
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct DiagonalTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


    template<typename MatrixType, unsigned int UpLo>
    struct TriangularTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::none;
      static constexpr bool is_triangular_adapter = false;

      template<TriangleType t, typename Arg>
      static constexpr auto make_triangular_matrix(Arg&& arg)
      {
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        return std::forward<Arg>(arg).nestedExpression().template triangularView<TriMode>();
      }
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int UpLo> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    template<typename MatrixType, unsigned int UpLo>
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (UpLo & Eigen::Upper) != 0 ? TriangleType::upper : TriangleType::lower;

      // make_hermitian_adapter not included because SelfAdjointView is already hermitian.

    };

  } // namespace interface


  /**
   * \brief Deduction guide for converting Eigen::SelfAdjointView to SelfAdjointMatrix
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_SelfAdjointView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_SelfAdjointView<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<nested_matrix_of_t<M>, hermitian_adapter_type_of_v<M>>;


  /**
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::SelfAdjointView<M, UpLo>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr TriangleType storage_triangle = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = storage_triangle, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = storage_triangle, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::SelfAdjointView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


  // ------- //
  //  Solve  //
  // ------- //

  namespace interface
  {

    template<typename Decomposition, typename RhsType>
    struct Dependencies<Eigen::Solve<Decomposition, RhsType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<const Decomposition&, const RhsType&>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).dec();
        else
          return std::forward<Arg>(arg).rhs();
      }

      // Eigen::Solve can never be self-contained.

    };


  // ----------- //
  //  Transpose  //
  // ----------- //

    template<typename MatrixType>
    struct Dependencies<Eigen::Transpose<MatrixType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename EGI::ref_selector<MatrixType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using M = equivalent_self_contained_t<MatrixType>;
        using N = Eigen::Transpose<M>;
        if constexpr (not std::is_lvalue_reference_v<typename EGI::ref_selector<M>::non_const_type>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename MatrixType>
    struct SingleConstant<Eigen::Transpose<MatrixType>> : SingleConstant<std::decay_t<MatrixType>>
    {
      SingleConstant(const Eigen::Transpose<MatrixType>& xpr) :
        SingleConstant<std::decay_t<MatrixType>> {xpr.nestedExpression()} {};
    };



    template<typename MatrixType>
    struct DiagonalTraits<Eigen::Transpose<MatrixType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType>
    struct TriangularTraits<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct TriangularTraits<Eigen::Transpose<MatrixType>, std::enable_if_t<triangular_matrix<MatrixType>>>
#endif
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (lower_triangular_matrix<MatrixType> ? TriangleType::upper : TriangleType::lower);
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>, std::enable_if_t<hermitian_matrix<MatrixType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };

  } // namespace interface

    // ---------------- //
    //  TriangularView  //
    // ---------------- //

  /**
   * \brief Deduction guide for converting Eigen::TriangularView to TriangularMatrix
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_TriangularView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_TriangularView<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<nested_matrix_of_t<M>, triangle_type_of_v<M>>;


  namespace interface
  {
    template<typename M, unsigned int Mode>
    struct Dependencies<Eigen::TriangularView<M, Mode>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename EGI::traits<Eigen::TriangularView<M, Mode>>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(TriangularMatrix {std::forward<Arg>(arg)});
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
    {
    private:

      template<int c, Likelihood b = Likelihood::definitely>
      struct C
      {
        using value_type = scalar_type_of_t<MatrixType>;
        static constexpr value_type value = c;
        static constexpr Likelihood status = b;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
      };

    public:

      const Eigen::TriangularView<MatrixType, Mode>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (diagonal_matrix<MatrixType> and (Mode & Eigen::ZeroDiag) != 0)
          return C<0>{};
        else
          return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr ((Mode & Eigen::ZeroDiag) == 0 and eigen_Identity<MatrixType>)
        {
          return C<1>{};
        }
        else if constexpr (((Mode & Eigen::UnitDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType, Likelihood::maybe>))))
        {
          return C<1, triangular_matrix<MatrixType> ? Likelihood::definitely : Likelihood::maybe>{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType, Likelihood::maybe>)))
        {
          return C<0, triangular_matrix<MatrixType> ? Likelihood::definitely : Likelihood::maybe>{};
        }
        else
        {
          return constant_diagonal_coefficient {xpr.nestedExpression()};
        }
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct DiagonalTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      static constexpr bool is_diagonal = ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>);
    };


    template<typename MatrixType, unsigned int Mode>
    struct TriangularTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      static constexpr TriangleType triangle_type = (Mode & Eigen::Upper) != 0 ?
        (lower_triangular_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::upper) :
        (upper_triangular_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::lower);

      static constexpr bool is_triangular_adapter = true;

      template<TriangleType t, typename Arg>
      static constexpr decltype(auto) make_triangular_matrix(Arg&& arg)
      {
        auto& n = std::forward<Arg>(arg).nestedExpression();
        return TriangularMatrix<decltype(n), TriangleType::diagonal> {n};
      }
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int Mode> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>>>
#endif
    {
      static constexpr bool is_hermitian = diagonal_matrix<MatrixType>;
      static constexpr TriangleType adapter_type = is_hermitian ? TriangleType::diagonal : TriangleType::none;

      template<TriangleType t, typename Arg>
      static constexpr auto make_hermitian_adapter(Arg&& arg)
      {
        static_assert(not hermitian_matrix<Arg>);
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        return std::forward<Arg>(arg).nestedExpression().template selfadjointView<TriMode>();
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
  template<typename M, unsigned int Mode>
  struct MatrixTraits<Eigen::TriangularView<M, Mode>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::TriangularView<M, Mode>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr TriangleType triangle_type = Mode & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = triangle_type, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::TriangularView<std::remove_reference_t<Arg>, Mode>(arg);
    }

  };


  // ------------- //
  //  VectorBlock  //
  // ------------- //

  namespace interface
  {

    template<typename VectorType, int Size>
    struct Dependencies<Eigen::VectorBlock<VectorType, Size>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename EGI::ref_selector<VectorType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::VectorBlock should always be converted to Matrix

    };


    template<typename VectorType, int Size>
    struct SingleConstant<Eigen::VectorBlock<VectorType, Size>>
    {
      const Eigen::VectorBlock<VectorType, Size>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr.nestedExpression()};
      }
    };


  // -------------- //
  //  VectorWiseOp  //
  // -------------- //

    template<typename ExpressionType, int Direction>
    struct IndexibleObjectTraits<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      static constexpr std::size_t max_indices = 2;
      using scalar_type = typename std::decay_t<Eigen::VectorwiseOp<ExpressionType, Direction>>::Scalar;
    };


    template<typename ExpressionType, int Direction, std::size_t N>
    struct IndexTraits<Eigen::VectorwiseOp<ExpressionType, Direction>, N> : IndexTraits<ExpressionType, N> {};


    template<typename ExpressionType, int Direction>
    struct Dependencies<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename ExpressionType::ExpressionTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg)._expression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::VectorwiseOp<equivalent_self_contained_t<ExpressionType>, Direction>;
        static_assert(self_contained<typename N::ExpressionTypeNested>,
          "This VectorWiseOp expression cannot be made self-contained");
        return N {make_self_contained(arg._expression())};
      }
    };


    template<typename ExpressionType, int Direction>
    struct SingleConstant<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      const Eigen::VectorwiseOp<ExpressionType, Direction>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr._expression()};
      }
    };

  } // namespace interface

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
