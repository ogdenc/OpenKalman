/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, HermitianAdapterType::lower>;


#ifdef __cpp_concepts
  template<HermitianAdapterType t = HermitianAdapterType::lower, native_eigen_matrix M>
#else
  template<HermitianAdapterType t = HermitianAdapterType::lower, typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
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
      static constexpr std::size_t max_indices =
        std::decay_t<T>::RowsAtCompileTime == 0 or std::decay_t<T>::ColsAtCompileTime == 0 ? 0 : 2;

      using scalar_type = typename std::decay_t<T>::Scalar;
    };


#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexTraits<T>
#else
    namespace detail
    {
    template<typename T>
    struct IndexTraits_Eigen_default
#endif
    {
    private:

      template<std::size_t N>
      static constexpr auto e_dim = N == 0 ? T::RowsAtCompileTime : T::ColsAtCompileTime;

    public:

      template<std::size_t N>
      static constexpr std::size_t dimension = e_dim<N> == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(e_dim<N>);

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> == dynamic_size)
        {
          if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return dimension<N>;
      }
    };
#ifndef __cpp_concepts
    } // namespace detail
#endif



#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      template<Eigen::Index...Is>
      using dense_type = std::conditional_t<native_eigen_array<T>,
        Eigen::Array<Scalar, Is...>, Eigen::Matrix<Scalar, Is...>>;

      template<std::size_t...Is>
      using writable_type = dense_type<(Is == dynamic_size ? Eigen::Dynamic : static_cast<Eigen::Index>(Is))...>;

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

      static constexpr bool is_writable = native_eigen_dense<T> and
        static_cast<bool>(Eigen::internal::traits<std::decay_t<T>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit));


      template<typename...D>
      static auto make_default(D&&...d)
      {
        using M = writable_type<dimension_size_of_v<D>...>;

        if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
          return M(static_cast<Eigen::Index>(get_dimension_size_of(d))...);
        else
          return M {};
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using M = writable_type<index_dimension_of_v<Arg, 0>, index_dimension_of_v<Arg, 1>>;

        if constexpr (eigen_DiagonalWrapper<Arg>)
        {
          // Note: Arg's nested matrix might not be a column vector.
          return M {OpenKalman::to_diagonal(OpenKalman::diagonal_of(std::forward<Arg>(arg)))};
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
#ifdef __cpp_concepts
      template<scalar_constant<CompileTimeStatus::unknown> C, typename...Ds> requires (sizeof...(Ds) == 2)
      static constexpr constant_matrix<CompileTimeStatus::unknown> auto
#else
      template<typename C, typename...Ds, std::enable_if_t<
        scalar_constant<C, CompileTimeStatus::unknown> and (sizeof...(Ds) == 2), int> = 0>
      static constexpr auto
#endif
      make_constant_matrix(C&& c, Ds&&...ds)
      {
        using N = dense_writable_matrix_t<T, Scalar, std::decay_t<Ds>...>;
        return N::Constant(static_cast<Eigen::Index>(get_dimension_size_of(std::forward<Ds>(ds)))..., std::forward<C>(c));
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

#ifndef __cpp_concepts
    template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    struct IndexTraits<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
      : detail::IndexTraits_Eigen_default<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {};
#endif


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
    struct IndexTraits<Eigen::ArrayWrapper<XprType>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<XprType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<XprType, b>;
    };


    template<typename XprType>
    struct Dependencies<Eigen::ArrayWrapper<XprType>>
    {
    private:

      using NestedXpr = typename Eigen::ArrayWrapper<XprType>::NestedExpressionType;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedXpr>;

      template<std::size_t i, typename Arg>
      static NestedXpr get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (std::is_lvalue_reference_v<NestedXpr>)
          return const_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
        else
          return static_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
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
    struct TriangularTraits<Eigen::ArrayWrapper<XprType>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


    template<typename XprType>
    struct Conversions<Eigen::ArrayWrapper<XprType>>
    {
      template<typename Arg>
      static constexpr decltype(auto) to_diagonal(Arg&& arg) { return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg))); }

      template<typename Arg>
      static constexpr decltype(auto) diagonal_of(Arg&& arg) { return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg))); }
    };


    // ------- //
    //  Block  //
    // ------- //

#ifndef __cpp_concepts
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct IndexTraits<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
      : detail::IndexTraits_Eigen_default<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> {};
#endif


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

#ifndef __cpp_concepts
    template<typename MatrixType, int DiagIndex>
    struct IndexTraits<Eigen::Diagonal<MatrixType, DiagIndex>>
      : detail::IndexTraits_Eigen_default<Eigen::Diagonal<MatrixType, DiagIndex>> {};
#endif


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
        if constexpr (constant_diagonal_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          if constexpr (DiagIndex == Eigen::DynamicIndex)
          {
            if (xpr.index() == 0)
              return constant_diagonal_coefficient{xpr.nestedExpression()}();
            else
              return scalar_type_of_t<MatrixType>{0};
          }
          else if constexpr (DiagIndex == 0)
            return constant_diagonal_coefficient{xpr.nestedExpression()};
          else
            return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
        }
        else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          return constant_coefficient{xpr.nestedExpression()};
        }
        else
        {
          return std::monostate{};
        }
      }
    };


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct IndexTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = SizeAtCompileTime == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(SizeAtCompileTime);

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (SizeAtCompileTime == Eigen::Dynamic) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(SizeAtCompileTime);
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = SizeAtCompileTime == 1 or (SizeAtCompileTime == Eigen::Dynamic and b == Likelihood::maybe);

      template<Likelihood b>
      static constexpr bool is_square = true;
    };


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
    struct TriangularTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_diagonal_adapter = true;
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


  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct Conversions<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
  {
    template<typename Arg>
    static decltype(auto)
    to_diagonal(Arg&& arg)
    {
      // In this case, arg will be one-by-one.
      if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
        throw std::logic_error {"Argument of to_diagonal must be 1-by-1"};

      using M = Eigen::Matrix<scalar_type_of_t<Arg>, 1, 1>;
      return M {std::forward<Arg>(arg).diagonal()};
    }


    template<typename Arg>
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg)
    {
      auto d {make_self_contained<Arg>(std::forward<Arg>(arg).diagonal())};

      if constexpr (std::is_lvalue_reference_v<Arg> or not has_dynamic_dimensions<Arg> or SizeAtCompileTime == Eigen::Dynamic)
        return d;
      else
        return untyped_dense_writable_matrix_t<decltype(d), Scalar, static_cast<std::size_t>(SizeAtCompileTime), 1> {std::move(d)};
    }
  };


  // ----------------- //
  //  DiagonalWrapper  //
  // ----------------- //

  namespace interface
  {
    template<typename DiagVectorType>
    struct IndexTraits<Eigen::DiagonalWrapper<DiagVectorType>>
    {
    private:

      static constexpr auto e_dim = Eigen::DiagonalWrapper<DiagVectorType>::RowsAtCompileTime;

    public:

      template<std::size_t N>
      static constexpr std::size_t dimension = has_dynamic_dimensions<DiagVectorType> ? dynamic_size :
        index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> == dynamic_size) return static_cast<std::size_t>(arg.rows());
        else return dimension<N>;
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<DiagVectorType, b>;

      template<Likelihood b>
      static constexpr bool is_square = true;
    };


    template<typename DiagVectorType>
    struct Dependencies<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename DiagVectorType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        decltype(auto) d = std::forward<Arg>(arg).diagonal();
        using D = decltype(d);
        using NCD = std::conditional_t<
          std::is_const_v<std::remove_reference_t<Arg>> or std::is_const_v<DiagVectorType>,
          D, std::conditional_t<std::is_lvalue_reference_v<D>, std::decay_t<D>&, std::decay_t<D>>>;
        return const_cast<NCD>(std::forward<decltype(d)>(d));
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename DiagVectorType>
    struct SingleConstant<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      const Eigen::DiagonalWrapper<DiagVectorType>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient {xpr.diagonal()};
      }
    };


    template<typename DiagVectorType>
    struct TriangularTraits<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_diagonal_adapter = true;
    };


    template<typename DiagVectorType>
    struct Conversions<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      template<typename Arg>
      static constexpr decltype(auto) to_diagonal(Arg&& arg)
      {
        // In this case, arg will be a one-by-one matrix.
        if constexpr (has_dynamic_dimensions<DiagVectorType>)
          if (get_index_dimension_of<0>(arg) + get_index_dimension_of<1>(arg) != 1) throw std::logic_error {
            "Argument of to_diagonal must have 1 element; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};

        return make_self_contained<Arg>(std::forward<Arg>(arg).diagonal());
      }


      template<typename Arg>
      static constexpr decltype(auto) diagonal_of(Arg&& arg)
      {
        using Scalar = scalar_type_of_t<Arg>;
        decltype(auto) diag {nested_matrix(std::forward<Arg>(arg))};
        using Diag = decltype(diag);
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr Eigen::Index rows = EigenTraits::RowsAtCompileTime;
        constexpr Eigen::Index cols = EigenTraits::ColsAtCompileTime;

        if constexpr (cols == 1 or cols == 0)
        {
          return std::forward<Diag>(diag);
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          return transpose(std::forward<Diag>(diag));
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          auto d {make_dense_writable_matrix_from(std::forward<Diag>(diag))};
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data())};
        }
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalWrapper.
   */
  template<typename V>
  struct MatrixTraits<Eigen::DiagonalWrapper<V>>
    : MatrixTraits<Eigen::Matrix<typename EGI::traits<std::decay_t<V>>::Scalar,
        V::SizeAtCompileTime, V::SizeAtCompileTime>> {};

} // namespace OpenKalman


namespace OpenKalman
{
  namespace interface
  {

    // ------------- //
    //  Homogeneous  //
    // ------------- //
    // \todo: Add. This is a child of Eigen::MatrixBase

#ifndef __cpp_concepts
    template<typename MatrixType,int Direction>
    struct IndexTraits<Eigen::Homogeneous<MatrixType, Direction>>
      : detail::IndexTraits_Eigen_default<Eigen::Homogeneous<MatrixType, Direction>> {};
#endif


#if EIGEN_VERSION_AT_LEAST(3,4,0)

    // ------------------------- //
    //  IndexedView (Eigen 3.4)  //
    // ------------------------- //

#ifndef __cpp_concepts
    template<typename XprType, typename RowIndices, typename ColIndices>
    struct IndexTraits<Eigen::IndexedView<XprType, RowIndices, ColIndices>>
      : detail::IndexTraits_Eigen_default<Eigen::IndexedView<XprType, RowIndices, ColIndices>> {};
#endif

#endif


    // --------- //
    //  Inverse  //
    // --------- //

#ifndef __cpp_concepts
    template<typename XprType>
    struct IndexTraits<Eigen::Inverse<XprType>> : detail::IndexTraits_Eigen_default<Eigen::Inverse<XprType>> {};
#endif


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

#ifndef __cpp_concepts
    template<typename PlainObjectType, int MapOptions, typename StrideType>
    struct IndexTraits<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
      : detail::IndexTraits_Eigen_default<Eigen::Map<PlainObjectType, MapOptions, StrideType>> {};
#endif


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

#ifndef __cpp_concepts
    template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
    struct IndexTraits<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
      : detail::IndexTraits_Eigen_default<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>> {};
#endif


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
    struct IndexTraits<Eigen::MatrixWrapper<XprType>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<XprType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<XprType, b>;
    };


    template<typename XprType>
    struct Dependencies<Eigen::MatrixWrapper<XprType>>
    {
    private:

      using NestedXpr = typename Eigen::MatrixWrapper<XprType>::NestedExpressionType;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedXpr>;

      template<std::size_t i, typename Arg>
      static NestedXpr get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (std::is_lvalue_reference_v<NestedXpr>)
          return const_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
        else
          return static_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::MatrixWrapper<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return make_self_contained(arg.nestedExpression()).matrix();
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
    struct TriangularTraits<Eigen::MatrixWrapper<XprType>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


    template<typename XprType>
    struct Conversions<Eigen::MatrixWrapper<XprType>>
    {
      template<typename Arg>
      static constexpr decltype(auto) to_diagonal(Arg&& arg) { return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg))); }

      template<typename Arg>
      static constexpr decltype(auto) diagonal_of(Arg&& arg) { return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg))); }
    };


    // ------------------- //
    //  PermutationMatrix  //
    // ------------------- //

#ifndef __cpp_concepts
    template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
    struct IndexTraits<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
      : detail::IndexTraits_Eigen_default<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>> {};
#endif


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

#ifndef __cpp_concepts
    template<typename IndicesType>
    struct IndexTraits<Eigen::PermutationWrapper<IndicesType>>
      : detail::IndexTraits_Eigen_default<Eigen::PermutationWrapper<IndicesType>> {};
#endif


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

#ifndef __cpp_concepts
    template<typename LhsType, typename RhsType, int Option>
    struct IndexTraits<Eigen::Product<LhsType, RhsType, Option>>
      : detail::IndexTraits_Eigen_default<Eigen::Product<LhsType, RhsType, Option>> {};
#endif


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
        if constexpr (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and
          constant_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
        {
          return scalar_constant_operation {std::multiplies<>{},
             constant_diagonal_coefficient{xpr.lhs()}, constant_coefficient{xpr.rhs()}};
        }
        else if constexpr (constant_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and
          constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
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

          if constexpr (zero_matrix<Arg1>) return constant_coefficient{xpr.lhs()};
          else if constexpr (zero_matrix<Arg2>) return constant_coefficient{xpr.rhs()};
          else if constexpr (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)
            return scalar_constant_operation {std::multiplies<>{},
              constant_diagonal_coefficient{xpr.lhs()},
              constant_coefficient{xpr.rhs()}};
          else if constexpr (constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
            return scalar_constant_operation {std::multiplies<>{},
              constant_coefficient{xpr.lhs()},
              constant_diagonal_coefficient{xpr.rhs()}};
          else if constexpr (dim == dynamic_size)
            return scalar_constant_operation {Op{},
              get_index_dimension_of<1>(xpr.lhs()),
              constant_coefficient{xpr.rhs()},
              constant_coefficient{xpr.lhs()}};
          else
            return scalar_constant_operation {Op{},
              std::integral_constant<std::size_t, dim>{},
              constant_coefficient{xpr.rhs()},
              constant_coefficient{xpr.lhs()}};
        }
      }
      
      constexpr auto get_constant_diagonal()
      {
        return scalar_constant_operation {std::multiplies<>{},
          constant_diagonal_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
      }
    };


    template<typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<Arg1, t, b> and triangular_matrix<Arg2, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


    /// A constant diagonal matrix times a self-adjoint matrix (or vice versa) is self-adjoint.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix<CompileTimeStatus::any, Likelihood::maybe> Arg1, hermitian_matrix<Likelihood::maybe> Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<Arg2, Likelihood::maybe>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


    /// A self-adjoint matrix times a constant-diagonal matrix is self-adjoint.
#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> Arg1, constant_diagonal_matrix<CompileTimeStatus::any, Likelihood::maybe> Arg2>
      requires (not constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (hermitian_matrix<Arg1, Likelihood::maybe> and constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe> and
      not constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


  // ----- //
  //  Ref  //
  // ----- //

#ifndef __cpp_concepts
    template<typename PlainObjectType, int Options, typename StrideType>
    struct IndexTraits<Eigen::Ref<PlainObjectType, Options, StrideType>>
      : detail::IndexTraits_Eigen_default<Eigen::Ref<PlainObjectType, Options, StrideType>> {};
#endif


    template<typename PlainObjectType, int Options, typename StrideType>
    struct Dependencies<Eigen::Ref<PlainObjectType, Options, StrideType>>
    {
      static constexpr bool has_runtime_parameters = false;
      // Ref is not self-contained in any circumstances.
    };

  } // namespace interface


  // ----------- //
  //  Replicate  //
  // ----------- //

  namespace interface
  {
    template<typename MatrixType, int RowFactor, int ColFactor>
    struct IndexTraits<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    {
    private:

      using T = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;

      template<std::size_t N>
      static constexpr auto dim = N == 0 ? T::RowsAtCompileTime : T::ColsAtCompileTime;

    public:

      template<std::size_t N>
      static constexpr std::size_t dimension =
        dim<N> == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(dim<N>);

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> == dynamic_size)
        {
          if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return dimension<N>;
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one =
        (b != Likelihood::definitely or (RowFactor == 1 and ColFactor == 1)) and
        (RowFactor == 1 or RowFactor == Eigen::Dynamic) and
        (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
        one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square =
        (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
        (RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic or
          ((RowFactor != ColFactor or square_matrix<MatrixType, b>) and
          (dynamic_dimension<MatrixType, 0> or RowFactor * index_dimension_of_v<MatrixType, 0> % ColFactor == 0) and
          (dynamic_dimension<MatrixType, 1> or ColFactor * index_dimension_of_v<MatrixType, 1> % RowFactor == 0))) and
        (has_dynamic_dimensions<MatrixType> or
          ((RowFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 0> * RowFactor % index_dimension_of_v<MatrixType, 1> == 0) and
          (ColFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 1> * ColFactor % index_dimension_of_v<MatrixType, 0> == 0)));
    };


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

      constexpr auto get_constant_diagonal()
      {
        if constexpr (RowFactor == 1 and ColFactor == 1)
        {
          return constant_diagonal_coefficient {xpr.nestedExpression()};
        }
        else if constexpr ((RowFactor == 1 or RowFactor == Eigen::Dynamic) and
          (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
          constant_diagonal_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          constant_diagonal_coefficient cd {xpr.nestedExpression()};
          return internal::ScalarConstant<Likelihood::maybe, std::decay_t<decltype(cd)>> {cd};
        }
        else return std::monostate {};
      }
    };


    template<typename MatrixType, int RowFactor, int ColFactor>
    struct TriangularTraits<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = RowFactor == 1 and ColFactor == 1 and triangular_matrix<MatrixType, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<
      hermitian_matrix<MatrixType, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


#if EIGEN_VERSION_AT_LEAST(3,4,0)

    // ---------------------- //
    //  Reshaped (Eigen 3.4)  //
    // ---------------------- //

    template<typename XprType, int Rows, int Cols, int Order>
    struct IndexTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    {
    private:

      static constexpr std::size_t xprtypeprod = has_dynamic_dimensions<XprType> ? dynamic_size :
        index_dimension_of_v<XprType, 0> * index_dimension_of_v<XprType, 1>;

      static constexpr std::size_t xprtypemax = std::max(
        dynamic_dimension<XprType, 0> ? 0 : index_dimension_of_v<XprType, 0>,
        dynamic_dimension<XprType, 1> ? 0 : index_dimension_of_v<XprType, 1>);

      template<std::size_t N>
      static constexpr auto dim = N == 0 ? Rows : Cols;

      template<std::size_t N>
      static constexpr std::size_t dimension_i =
        dim<N> != Eigen::Dynamic ? static_cast<std::size_t>(dim<N>) :
        dim<N == 1 ? 0 : 1> == Eigen::Dynamic or dim<N == 1 ? 0 : 1> == 0 ? dynamic_size :
        dim<N == 1 ? 0 : 1> == index_dimension_of_v<XprType, 0> ? index_dimension_of_v<XprType, 1> :
        dim<N == 1 ? 0 : 1> == index_dimension_of_v<XprType, 1> ? index_dimension_of_v<XprType, 0> :
        xprtypeprod != dynamic_size and xprtypeprod % dim<N == 1 ? 0 : 1> == 0 ? xprtypeprod / dim<N == 1 ? 0 : 1> :
        dynamic_size;

    public:

      template<std::size_t N>
      static constexpr std::size_t dimension = dimension_i<N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> == dynamic_size)
        {
          if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return dimension<N>;
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one =
        (Rows == 1 and Cols == 1 and one_by_one_matrix<XprType, Likelihood::maybe>) or
        ((Rows == 1 or Rows == Eigen::Dynamic) and (Cols == 1 or Cols == Eigen::Dynamic) and one_by_one_matrix<XprType, b>);

      template<Likelihood b>
      static constexpr bool is_square =
        (b != Likelihood::definitely or (Rows != Eigen::Dynamic and Cols != Eigen::Dynamic) or
          ((Rows != Eigen::Dynamic or Cols != Eigen::Dynamic) and number_of_dynamic_indices_v<XprType> <= 1)) and
        (Rows == Eigen::Dynamic or Cols == Eigen::Dynamic or Rows == Cols) and
        (xprtypeprod == dynamic_size or (
          are_within_tolerance(xprtypeprod, constexpr_sqrt(xprtypeprod) * constexpr_sqrt(xprtypeprod)) and
          (Rows == Eigen::Dynamic or Rows * Rows == xprtypeprod) and
          (Cols == Eigen::Dynamic or Cols * Cols == xprtypeprod))) and
        (Rows == Eigen::Dynamic or xprtypemax == 0 or (Rows * Rows) % xprtypemax == 0) and
        (Cols == Eigen::Dynamic or xprtypemax == 0 or (Cols * Cols) % xprtypemax == 0);
    };


    namespace detail
    {
      template<typename XprType, int Rows, int Cols, int Order, bool HasDirectAccess>
      struct ReshapedNested { using type = typename Eigen::Reshaped<XprType, Rows, Cols, Order>::MatrixTypeNested; };

      template<typename XprType, int Rows, int Cols, int Order>
      struct ReshapedNested<XprType, Rows, Cols, Order, true>
      {
        using type = typename Eigen::internal::ref_selector<XprType>::non_const_type;
      };
    }


    template<typename XprType, int Rows, int Cols, int Order>
    struct Dependencies<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    {
    private:
      using R = Eigen::Reshaped<XprType, Rows, Cols, Order>;

      static constexpr bool HasDirectAccess = Eigen::internal::traits<R>::HasDirectAccess;

      using Nested_t = typename detail::ReshapedNested<XprType, Rows, Cols, Order, HasDirectAccess>::type;

    public:

      static constexpr bool has_runtime_parameters = HasDirectAccess ? Rows == Eigen::Dynamic or Cols == Eigen::Dynamic : false;

      using type = std::tuple<Nested_t>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename XprType, int Rows, int Cols, int Order>
    struct SingleConstant<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    {
      const Eigen::Reshaped<XprType, Rows, Cols, Order>& xpr;

      constexpr auto get_constant()
      {
        return constant_coefficient {xpr.nestedExpression()};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr (
          (Rows != Eigen::Dynamic and (Rows == XprType::RowsAtCompileTime or Rows == XprType::ColsAtCompileTime or Rows == Cols)) or
          (Cols != Eigen::Dynamic and (Cols == XprType::RowsAtCompileTime or Cols == XprType::ColsAtCompileTime)))
        {
          return constant_diagonal_coefficient {xpr.nestedExpression()};
        }
        else if constexpr (((Rows == Eigen::Dynamic and Cols == Eigen::Dynamic) or
          (XprType::RowsAtCompileTime == Eigen::Dynamic and XprType::ColsAtCompileTime == Eigen::Dynamic)) and
          constant_diagonal_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          constant_diagonal_coefficient cd {xpr.nestedExpression()};
          return internal::ScalarConstant<Likelihood::maybe, std::decay_t<decltype(cd)>> {cd};
        }
        else return std::monostate{};
      }
    };


    template<typename XprType, int Rows, int Cols, int Order>
    struct TriangularTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<XprType, t, b> and
        (Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
          Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
          (Rows != Eigen::Dynamic and Rows == Cols));

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> XprType, int Rows, int Cols, int Order>
    struct HermitianTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>>
#else
    template<typename XprType, int Rows, int Cols, int Order>
    struct HermitianTraits<Eigen::Reshaped<XprType, Rows, Cols, Order>, std::enable_if_t<
      hermitian_matrix<XprType, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian =
        Rows == index_dimension_of_v<XprType, 0> or Rows == index_dimension_of_v<XprType, 1> or
          Cols == index_dimension_of_v<XprType, 1> or Cols == index_dimension_of_v<XprType, 0> or
          (Rows != Eigen::Dynamic and Rows == Cols);
    };

#endif // EIGEN_VERSION_AT_LEAST(3,4,0)


  // --------- //
  //  Reverse  //
  // --------- //

    template<typename MatrixType, int Direction>
    struct IndexTraits<Eigen::Reverse<MatrixType, Direction>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<MatrixType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;
    };


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
    struct TriangularTraits<Eigen::Reverse<MatrixType, Direction>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<MatrixType,
          t == TriangleType::upper ? TriangleType::lower :
          t == TriangleType::lower ? TriangleType::upper : t, b> and
        (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>);

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> MatrixType, int Direction> requires
      (Direction == Eigen::BothDirections) or (one_by_one_matrix<MatrixType, Likelihood::maybe>)
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<hermitian_matrix<MatrixType, Likelihood::maybe> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType, Likelihood::maybe>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


    // -------- //
    //  Select  //
    // -------- //

    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct IndexTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension =
        not dynamic_dimension<ConditionMatrixType, N> ? index_dimension_of_v<ConditionMatrixType, N> :
        not dynamic_dimension<ThenMatrixType, N> ? index_dimension_of_v<ThenMatrixType, N> :
        not dynamic_dimension<ElseMatrixType, N> ? index_dimension_of_v<ElseMatrixType, N> :
        dynamic_size;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> != dynamic_size) return dimension<N>;
        else return get_index_dimension_of<N>(arg.conditionMatrix());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one =
        one_by_one_matrix<ConditionMatrixType, Likelihood::maybe> and one_by_one_matrix<ThenMatrixType, Likelihood::maybe> and one_by_one_matrix<ElseMatrixType, Likelihood::maybe> and
        (b != Likelihood::definitely or one_by_one_matrix<ConditionMatrixType, b> or one_by_one_matrix<ThenMatrixType, b> or one_by_one_matrix<ElseMatrixType, b>);
    };


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
        if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
            return constant_coefficient{xpr.thenMatrix()};
          else
            return constant_coefficient{xpr.elseMatrix()};
        }
        else if constexpr (constant_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
          constant_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          if constexpr (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
            return constant_coefficient{xpr.thenMatrix()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        if constexpr (constant_matrix<ConditionMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          if constexpr (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
            return constant_diagonal_coefficient{xpr.thenMatrix()};
          else
            return constant_diagonal_coefficient{xpr.elseMatrix()};
        }
        else if constexpr (constant_diagonal_matrix<ThenMatrixType, CompileTimeStatus::any, Likelihood::maybe> and
          constant_diagonal_matrix<ElseMatrixType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          if constexpr (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
            return constant_diagonal_coefficient{xpr.thenMatrix()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix<CompileTimeStatus::known, Likelihood::maybe> ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>>>
#endif
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular =
        (triangular_matrix<ThenMatrixType, t, b> and constant_coefficient_v<ConditionMatrixType>) or
        (triangular_matrix<ElseMatrixType, t, b> and not constant_coefficient_v<ConditionMatrixType>);

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<constant_matrix<CompileTimeStatus::known, Likelihood::maybe> ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
      (hermitian_matrix<ThenMatrixType, Likelihood::maybe> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
      (hermitian_matrix<ElseMatrixType, Likelihood::maybe> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
      constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe> and
      ((hermitian_matrix<ThenMatrixType, Likelihood::maybe> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
       (hermitian_matrix<ElseMatrixType, Likelihood::maybe> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> ConditionMatrixType, hermitian_matrix<Likelihood::maybe> ThenMatrixType,
        hermitian_matrix<Likelihood::maybe> ElseMatrixType> requires
      (not constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>)
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<hermitian_matrix<ConditionMatrixType, Likelihood::maybe> and hermitian_matrix<ThenMatrixType, Likelihood::maybe> and
        hermitian_matrix<ElseMatrixType, Likelihood::maybe> and
        (not constant_matrix<ConditionMatrixType, CompileTimeStatus::known, Likelihood::maybe>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


  // ----------------- //
  //  SelfAdjointView  //
  // ----------------- //

    template<typename MatrixType, unsigned int UpLo>
    struct IndexTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<MatrixType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;
    };


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
        constexpr auto t = hermitian_adapter_type_of_v<Arg>;
        return SelfAdjointMatrix<equivalent_self_contained_t<M>, t> {std::forward<Arg>(arg)};
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstant<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      const Eigen::SelfAdjointView<MatrixType, UpLo>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (not complex_number<scalar_type_of_t<MatrixType>>)
          return constant_coefficient{xpr.nestedExpression()};
        else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::known, Likelihood::maybe>)
        {
          if constexpr (real_axis_number<constant_coefficient<MatrixType>>)
            return constant_coefficient{xpr.nestedExpression()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        if constexpr (eigen_Identity<MatrixType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 1>{};
        else return constant_diagonal_coefficient {xpr.nestedExpression()};
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct TriangularTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = diagonal_matrix<MatrixType, b>;

      static constexpr bool is_triangular_adapter = false;

      template<TriangleType t, typename Arg>
      static constexpr auto make_triangular_matrix(Arg&& arg)
      {
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        if constexpr (TriMode == UpLo)
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().template triangularView<TriMode>());
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().adjoint().template triangularView<TriMode>());
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
      static constexpr HermitianAdapterType adapter_type =
        (UpLo & Eigen::Upper) != 0 ? HermitianAdapterType::upper : HermitianAdapterType::lower;

      // make_hermitian_adapter not included because SelfAdjointView is already hermitian if square.

    };


    template<typename MatrixType, unsigned int UpLo>
    struct Conversions<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
          // In this case, arg will be a one-by-one matrix.
          if constexpr (has_dynamic_dimensions<Arg>)
            if (get_index_dimension_of<0>(arg) != 1 or get_index_dimension_of<1>(arg) != 1) throw std::logic_error {
            "Argument of to_diagonal must be 1-by-1"};

          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression());
      }


      template<typename Arg>
      static constexpr decltype(auto)
      diagonal_of(Arg&& arg)
      {
        if constexpr (not square_matrix<Arg>) if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg))
          throw std::logic_error {"Argument of diagonal_of must be a square matrix; instead it has " +
            std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

        // Note: we assume that the nested matrix reference is not dangling.
        return OpenKalman::diagonal_of(std::forward<Arg>(arg).nestedExpression());
      }
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

    static constexpr HermitianAdapterType storage_triangle = UpLo & Eigen::Upper ? HermitianAdapterType::upper : HermitianAdapterType::lower;

  public:

    template<HermitianAdapterType storage_triangle = storage_triangle, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = storage_triangle ==
      HermitianAdapterType::upper ? TriangleType::upper : TriangleType::lower, std::size_t dim = rows>
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

} // namespace OpenKalman


  // ------- //
  //  Solve  //
  // ------- //

namespace OpenKalman
{
  namespace interface
  {

#ifndef __cpp_concepts
    template<typename Decomposition, typename RhsType>
    struct IndexTraits<Eigen::Solve<Decomposition, RhsType>>
      : detail::IndexTraits_Eigen_default<Eigen::Ref<Eigen::Solve<Decomposition, RhsType>>> {};
#endif


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
    struct IndexTraits<Eigen::Transpose<MatrixType>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<MatrixType, N == 0 ? 1 : 0>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N == 0 ? 1 : 0>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;
    };


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
    struct TriangularTraits<Eigen::Transpose<MatrixType>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = diagonal_matrix<MatrixType, b> or
        (t == TriangleType::lower and triangular_matrix<MatrixType, TriangleType::upper, b>) or
        (t == TriangleType::upper and triangular_matrix<MatrixType, TriangleType::lower, b>);

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>, std::enable_if_t<hermitian_matrix<MatrixType, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian = true;
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
    template<typename MatrixType, unsigned int Mode>
    struct IndexTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<MatrixType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;
    };


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
        return TriangularMatrix<equivalent_self_contained_t<M>, triangle_type_of_v<Arg>> {std::forward<Arg>(arg)};
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
    {
      const Eigen::TriangularView<MatrixType, Mode>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (zero_matrix<MatrixType> or ((Mode & Eigen::ZeroDiag) != 0 and diagonal_matrix<MatrixType, Likelihood::maybe>))
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
        else
          return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        constexpr auto b = has_dynamic_dimensions<MatrixType> ? Likelihood::maybe : Likelihood::definitely;

        if constexpr (not square_matrix<MatrixType, Likelihood::maybe>)
        {
          return std::monostate{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) == 0 and eigen_Identity<MatrixType>)
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr (((Mode & Eigen::UnitDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>))))
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>)))
        {
          return internal::ScalarConstant<b, Scalar, 0>{};
        }
        else
        {
          return constant_diagonal_coefficient {xpr.nestedExpression()};
        }
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct TriangularTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular =
        (t == TriangleType::lower and ((Mode & Eigen::Lower) != 0 or triangular_matrix<MatrixType, TriangleType::lower, b>)) or
        (t == TriangleType::upper and ((Mode & Eigen::Upper) != 0 or triangular_matrix<MatrixType, TriangleType::upper, b>)) or
        (t == TriangleType::diagonal and triangular_matrix<MatrixType, (Mode & Eigen::Lower) ? TriangleType::upper : TriangleType::lower, b>) or
        (t == TriangleType::any and square_matrix<MatrixType, b>);

      static constexpr bool is_triangular_adapter = true;

      // make_triangular_matrix not included because TriangularView is already triangular if square.
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

      template<HermitianAdapterType t, typename Arg>
      static constexpr auto make_hermitian_adapter(Arg&& arg)
      {
        constexpr auto HMode = t == HermitianAdapterType::upper ? Eigen::Upper : Eigen::Lower;
        constexpr auto TriMode = (Mode & Eigen::Upper) != 0 ? Eigen::Upper : Eigen::Lower;
        if constexpr (HMode == TriMode)
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().template selfadjointView<HMode>());
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().adjoint().template selfadjointView<HMode>());
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct Conversions<Eigen::TriangularView<MatrixType, Mode>>
    {
      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
          // In this case, arg will be a one-by-one matrix.
          if constexpr (has_dynamic_dimensions<Arg>)
            if (get_index_dimension_of<0>(arg) != 1 or get_index_dimension_of<1>(arg) != 1) throw std::logic_error {
            "Argument of to_diagonal must be 1-by-1"};

          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression());
      }


      template<typename Arg>
      static constexpr decltype(auto)
      diagonal_of(Arg&& arg)
      {
        if constexpr (not square_matrix<Arg>) if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg))
          throw std::logic_error {"Argument of diagonal_of must be a square matrix; instead it has " +
            std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

        // Note: we assume that the nested matrix reference is not dangling.
        return OpenKalman::diagonal_of(std::forward<Arg>(arg).nestedExpression());
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

    template<HermitianAdapterType storage_triangle =
      triangle_type == TriangleType::upper ? HermitianAdapterType ::upper : HermitianAdapterType ::lower, std::size_t dim = rows>
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

#ifndef __cpp_concepts
    template<typename VectorType, int Size>
    struct IndexTraits<Eigen::VectorBlock<VectorType, Size>>
      : detail::IndexTraits_Eigen_default<Eigen::Ref<Eigen::VectorBlock<VectorType, Size>>> {};
#endif


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


    template<typename ExpressionType, int Direction>
    struct IndexTraits<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      template<std::size_t N>
      static constexpr std::size_t dimension = index_dimension_of_v<ExpressionType, N>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<N>(arg._expression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<ExpressionType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<ExpressionType, b>;
    };


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
