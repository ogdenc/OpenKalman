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
 * \internal
 * \file
 * \brief Native Eigen3 evaluators for Eigen3 extensions
 */

#ifndef OPENKALMAN_EIGEN3_NATIVE_EVALUATORS_HPP
#define OPENKALMAN_EIGEN3_NATIVE_EVALUATORS_HPP

#include <complex>

namespace Eigen::internal
{
  using namespace OpenKalman;

  // ---------------------- //
  //  Eigen3::EigenWrapper  //
  // ---------------------- //

  template<typename ArgType>
  struct evaluator<OpenKalman::Eigen3::EigenWrapper<ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Eigen3::EigenWrapper<ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& m) : Base {m.wrapped_expression} {}
  };


  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

  template<typename PatternMatrix, auto...constant>
  struct evaluator<ConstantAdapter<PatternMatrix, constant...>>
    : evaluator_base<ConstantAdapter<PatternMatrix, constant...>>
  {
    using Scalar = scalar_type_of_t<PatternMatrix>;
    using XprType = ConstantAdapter<PatternMatrix, constant...>;
    using M = Eigen3::eigen_matrix_t<Scalar, index_dimension_of_v<PatternMatrix, 0>, index_dimension_of_v<PatternMatrix, 1>>;

    enum {
      CoeffReadCost = 0,
      Flags = NoPreferredStorageOrderBit | LinearAccessBit | (traits<M>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      Alignment = AlignedMax
    };

    explicit constexpr evaluator(const XprType&) {}

    constexpr Scalar coeff(Index row, Index col) const
    {
      return Scalar {constant...};
    }

    constexpr Scalar coeff(Index row) const
    {
      return Scalar {constant...};
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      return internal::pset1<PacketType>(Scalar {constant...});
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row) const
    {
      return internal::pset1<PacketType>(Scalar {constant...});
    }

  };


  // ------------------- //
  //  SelfAdjointMatrix  //
  // ------------------- //

  template<typename ArgType, OpenKalman::TriangleType storage_triangle>
  struct evaluator<SelfAdjointMatrix<ArgType, storage_triangle>>
    : evaluator_base<SelfAdjointMatrix<ArgType, storage_triangle>>
  {
    using XprType = SelfAdjointMatrix<ArgType, storage_triangle>;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    using CoeffReturnType = typename XprType::CoeffReturnType;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };


    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_matrix()) {}


    auto& coeffRef(Index row, Index col)
    {
      static_assert(storage_triangle != TriangleType::diagonal or one_by_one_matrix<ArgType>,
        "Reference to element is not available for a diagonal SelfAdjointMatrix");

      static_assert(not OpenKalman::complex_number<Scalar>,
        "Reference to element is not available for a complex SelfAdjointMatrix");

      if constexpr (storage_triangle == TriangleType::upper)
      {
        if (row > col)
          return m_argImpl.coeffRef(col, row);
      }
      else if constexpr (storage_triangle == TriangleType::lower)
      {
        if (row < col)
          return m_argImpl.coeffRef(col, row);
      }

      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      static_assert(one_by_one_matrix<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one SelfAdjointMatrix");

      static_assert(not OpenKalman::complex_number<Scalar>,
        "Reference to element is not available for a complex SelfAdjointMatrix");

      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (storage_triangle == TriangleType::diagonal)
      {
        if (row != col)
        {
          if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
          {
            static std::remove_reference_t<CoeffReturnType> dummy = 0;
            return dummy;
          }
          else return Scalar {0};
        }
      }
      else if constexpr (storage_triangle == TriangleType::upper)
      {
        if (row > col) return OpenKalman::internal::constexpr_conj(m_argImpl.coeff(col, row));
      }
      else if constexpr (storage_triangle == TriangleType::lower)
      {
        if (row < col) return OpenKalman::internal::constexpr_conj(m_argImpl.coeff(col, row));
      }

      if (row == col)
        return OpenKalman::internal::constexpr_real(m_argImpl.coeff(row, col));
      else
        return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      static_assert(storage_triangle == TriangleType::diagonal or one_by_one_matrix<ArgType>,
        "Linear (single index) element access is only available for diagonal SelfAdjointMatrix");

      return m_argImpl.coeff(i);
    }

  protected:

    NestedEvaluator m_argImpl;
  };


  // ------------------ //
  //  TriangularMatrix  //
  // ------------------ //

  template<typename ArgType, OpenKalman::TriangleType triangle_type>
  struct evaluator<TriangularMatrix<ArgType, triangle_type>>
    : evaluator_base<TriangularMatrix<ArgType, triangle_type>>
  {
    using XprType = TriangularMatrix<ArgType, triangle_type>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };

    //static constexpr bool is_row_major = static_cast<bool>(NestedEvaluator::Flags & RowMajorBit);


    explicit evaluator(const XprType& m_arg) : m_argImpl {m_arg.nested_matrix()} {}


    auto& coeffRef(Index row, Index col)
    {
      static Scalar dummy;
      if constexpr(triangle_type == TriangleType::diagonal) {if (row != col) { dummy = 0; return dummy; }}
      else if constexpr(triangle_type == TriangleType::upper) {if (row > col) { dummy = 0; return dummy; }}
      else if constexpr(triangle_type == TriangleType::lower) {if (row < col) { dummy = 0; return dummy; }}
      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
      {
        static std::remove_reference_t<CoeffReturnType> dummy = 0;
        if constexpr(triangle_type == TriangleType::diagonal) {if (row != col) return dummy;}
        else if constexpr(triangle_type == TriangleType::upper) {if (row > col) return dummy;}
        else if constexpr(triangle_type == TriangleType::lower) {if (row < col) return dummy;}
      }
      else
      {
        if constexpr(triangle_type == TriangleType::diagonal) {if (row != col) return Scalar(0);}
        else if constexpr(triangle_type == TriangleType::upper) {if (row > col) return Scalar(0);}
        else if constexpr(triangle_type == TriangleType::lower) {if (row < col) return Scalar(0);}
      }
      return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      return m_argImpl.coeff(i);
    }

  protected:

    NestedEvaluator m_argImpl;

  };


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //

  template<typename ArgType>
  struct evaluator<OpenKalman::DiagonalMatrix<ArgType>>
    : evaluator_base<OpenKalman::DiagonalMatrix<ArgType>>
  {
    using XprType = OpenKalman::DiagonalMatrix<ArgType>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_matrix()) {}

    auto& coeffRef(Index row, Index col)
    {
      if (row == col)
        return m_argImpl.coeffRef(row);
      else
      {
        static Scalar dummy;
        dummy = 0;
        return dummy;
      }
    }

    CoeffReturnType coeff(Index row, Index col) const
    {
      if (row == col)
        return m_argImpl.coeff(row);
      else
      {
        if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
        {
          static std::remove_reference_t<CoeffReturnType> dummy = 0;
          return dummy;
        }
        else
          return Scalar(0);
      }
    }

  protected:

    NestedEvaluator m_argImpl;
  };


  // --------------------------------------- //
  //  Base classes for euclidean evaluators  //
  // --------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<index_descriptor TypedIndex, typename XprType, typename Nested, typename NestedEvaluator>
#else
    template<typename TypedIndex, typename XprType, typename Nested, typename NestedEvaluator, typename = void>
#endif
    struct Evaluator_EuclideanExpr_Base : evaluator_base<XprType>
    {
      using CoeffReturnType = typename traits<Nested>::Scalar;

      enum
      {
        Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
          (~(traits<Nested>::ColsAtCompileTime == 1 ? 0 : LinearAccessBit)),
        Alignment = NestedEvaluator::Alignment
      };

      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : m_argImpl {t} {}

    protected:

      NestedEvaluator m_argImpl;
    };


#ifdef __cpp_concepts
    template<euclidean_index_descriptor TypedIndex, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<TypedIndex, XprType, Nested, NestedEvaluator>
#else
    template<typename TypedIndex, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<TypedIndex, XprType, Nested, NestedEvaluator,
      std::enable_if_t<euclidean_index_descriptor<TypedIndex>>>
#endif
      : NestedEvaluator
    {
      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : NestedEvaluator {t} {}
    };

  }


  // ----------------- //
  //  ToEuclideanExpr  //
  // ----------------- //

  /**
   * \internal
   * \brief Evaluator for ToEuclideanExpr
   * \tparam Coeffs TypedIndex types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<ToEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, ToEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = ToEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        euclidean_index_descriptor<Coeffs> ? 0 :
        (int) Eigen::internal::functor_traits<Eigen::internal::scalar_sin_op<Scalar>>::Cost +
          (int) Eigen::internal::functor_traits<Eigen::internal::scalar_cos_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_matrix()}, i_descriptor {get_dimensions_of<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return to_euclidean_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return to_euclidean_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }

  };


  // ------------------- //
  //  FromEuclideanExpr  //
  // ------------------- //

  /**
   * \internal
   * \brief General evaluator for FromEuclideanExpr
   * \tparam Coeffs TypedIndex types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<FromEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, FromEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = FromEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        euclidean_index_descriptor<Coeffs> ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_atan_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_matrix()}, i_descriptor {get_dimensions_of<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return i_descriptor.from_euclidean_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return from_euclidean_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }
  };


  /**
   * \internal
   * \brief Specialized evaluator for FromEuclideanExpr that has a nested ToEuclideanExpr.
   * \details This amounts to wrapping angles.
   * \tparam Coeffs TypedIndex types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<
    FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>>
      : detail::Evaluator_EuclideanExpr_Base<Coeffs,
        FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>,
        std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost
    };

    template<typename Arg>
    explicit evaluator(const Arg& t) : Base {t.nested_matrix().nested_matrix()}, i_descriptor {get_dimensions_of<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return wrap_get_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (euclidean_index_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return wrap_get_element(i_descriptor, g, (std::size_t) row, 0);
      }
    }
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    FromEuclideanExpr<Coeffs, const ToEuclideanExpr<Coeffs, ArgType>>>
    : evaluator<FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<FromEuclideanExpr<Coeffs,
      ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<FromEuclideanExpr<Coeffs,
      ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    FromEuclideanExpr<Coeffs, const ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<FromEuclideanExpr<Coeffs, ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<FromEuclideanExpr<Coeffs,
      ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  // ------ //
  //  Mean  //
  // ------ //

  template<typename TypedIndex, typename ArgType>
  struct evaluator<OpenKalman::Mean<TypedIndex, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Mean<TypedIndex, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    enum
    {
      Flags = (euclidean_index_descriptor<TypedIndex> ? Base::Flags : Base::Flags & ~LvalueBit),
    };
    explicit evaluator(const XprType& m) : Base(m.nested_matrix()) {}
  };


  // -------- //
  //  Matrix  //
  // -------- //

  template<typename RowCoeffs, typename ColCoeffs, typename ArgType>
  struct evaluator<OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& m) : Base {m.nested_matrix()} {}
  };


  // --------------- //
  //  EuclideanMean  //
  // --------------- //

  template<typename Coeffs, typename ArgType>
  struct evaluator<OpenKalman::EuclideanMean<Coeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::EuclideanMean<Coeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    explicit evaluator(const XprType& expression) : Base {expression.nested_matrix()} {}
  };


  // ------------ //
  //  Covariance  //
  // ------------ //

  template<typename TypedIndex, typename ArgType>
  struct evaluator<OpenKalman::Covariance<TypedIndex, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::Covariance<TypedIndex, ArgType>&>()))>>
  {
    using XprType = OpenKalman::Covariance<TypedIndex, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::hermitian_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::to_covariance_nestable(m_arg)) {}
  };


  // ---------------------- //
  //  SquareRootCovariance  //
  // ---------------------- //

  template<typename TypedIndex, typename ArgType>
  struct evaluator<OpenKalman::SquareRootCovariance<TypedIndex, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::SquareRootCovariance<TypedIndex, ArgType>&>()))>>
  {
    using XprType = OpenKalman::SquareRootCovariance<TypedIndex, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::triangular_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::to_covariance_nestable(m_arg)) {}
  };


} // namespace Eigen::internal

#endif //OPENKALMAN_EIGEN3_NATIVE_EVALUATORS_HPP
