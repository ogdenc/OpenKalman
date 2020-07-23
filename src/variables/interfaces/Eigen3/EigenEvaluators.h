/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENEVALUATORS_H
#define OPENKALMAN_EIGENEVALUATORS_H

namespace Eigen::internal
{
  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::Mean<Coefficients, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Mean<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    enum
    {
      Flags = (Coefficients::axes_only ? Base::Flags : Base::Flags & ~LvalueBit),
    };
    explicit evaluator(const XprType& m) : Base(m.base_matrix()) {}
  };


  template<typename RowCoeffs, typename ColCoeffs, typename ArgType>
  struct evaluator<OpenKalman::TypedMatrix<RowCoeffs, ColCoeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::TypedMatrix<RowCoeffs, ColCoeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& m) : Base {m.base_matrix()} {}
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<OpenKalman::EuclideanMean<Coeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = OpenKalman::EuclideanMean<Coeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& expression) : Base {expression.base_matrix()} {}
  };


  template<typename ArgType, OpenKalman::TriangleType storage_triangle>
  struct evaluator<OpenKalman::EigenSelfAdjointMatrix<ArgType, storage_triangle>>
    : evaluator_base<OpenKalman::EigenSelfAdjointMatrix<ArgType, storage_triangle>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = OpenKalman::EigenSelfAdjointMatrix<ArgType, storage_triangle>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(base_matrix(m_arg)) {}

    auto& coeffRef(Index row, Index col)
    {
      if constexpr(storage_triangle == OpenKalman::TriangleType::diagonal)
      {
        if (row != col)
        {
          static Scalar dummy;
          dummy = 0;
          return dummy;
        }
        else
          return m_argImpl.coeffRef(col, row);
      }
      else if constexpr(storage_triangle == OpenKalman::TriangleType::upper)
      {
        if (row > col)
          return m_argImpl.coeffRef(col, row);
        else
          return m_argImpl.coeffRef(row, col);
      }
      else if constexpr(storage_triangle == OpenKalman::TriangleType::lower)
      {
        if (row < col)
          return m_argImpl.coeffRef(col, row);
        else
          return m_argImpl.coeffRef(row, col);
      }
      else
      {
        return m_argImpl.coeffRef(row, col);
      }
    }

    auto& coeffRef(Index i)
    {
      return m_argImpl.coeffRef(i);
    }

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr(storage_triangle == OpenKalman::TriangleType::diagonal)
      {
        if (row != col)
        {
          if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
          {
            static Scalar dummy;
            dummy = 0;
            return dummy;
          }
          else
            return Scalar(0);
        }
        else
          return m_argImpl.coeff(col, row);
      }
      else if constexpr(storage_triangle == OpenKalman::TriangleType::upper)
      {
        if (row > col)
          return m_argImpl.coeff(col, row);
        else
          return m_argImpl.coeff(row, col);
      }
      else if constexpr(storage_triangle == OpenKalman::TriangleType::lower)
      {
        if (row < col)
          return m_argImpl.coeff(col, row);
        else
          return m_argImpl.coeff(row, col);
      }
      else
      {
        return m_argImpl.coeff(row, col);
      }
    }

    CoeffReturnType coeff(Index i) const
    {
      return m_argImpl.coeff(i);
    }

  protected:
    NestedEvaluator m_argImpl;
  };


  template<typename ArgType, OpenKalman::TriangleType triangle_type>
  struct evaluator<OpenKalman::EigenTriangularMatrix<ArgType, triangle_type>>
    : evaluator_base<OpenKalman::EigenTriangularMatrix<ArgType, triangle_type>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = OpenKalman::EigenTriangularMatrix<ArgType, triangle_type>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(base_matrix(m_arg)) {}

    auto& coeffRef(Index row, Index col)
    {
      if constexpr(triangle_type == OpenKalman::TriangleType::diagonal)
      {
        if (row != col)
        {
          static Scalar dummy;
          dummy = 0;
          return dummy;
        }
        else
          return m_argImpl.coeffRef(col, row);
      }
      else if constexpr(triangle_type == OpenKalman::TriangleType::upper)
      {
        if (row > col)
        {
          static Scalar dummy;
          dummy = 0;
          return dummy;
        }
        else
          return m_argImpl.coeffRef(row, col);
      }
      else if constexpr(triangle_type == OpenKalman::TriangleType::lower)
      {
        if (row < col)
        {
          static Scalar dummy;
          dummy = 0;
          return dummy;
        }
        else
          return m_argImpl.coeffRef(row, col);
      }
      else
      {
        return m_argImpl.coeffRef(row, col);
      }
    }

    auto& coeffRef(Index i)
    {
      return m_argImpl.coeffRef(i);
    }

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr(triangle_type == OpenKalman::TriangleType::diagonal)
      {
        if (row != col)
        {
          if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
          {
            static Scalar dummy;
            dummy = 0;
            return dummy;
          }
          else
            return Scalar(0);
        }
        else
          return m_argImpl.coeff(col, row);
      }
      else if constexpr(triangle_type == OpenKalman::TriangleType::upper)
      {
        if (row > col)
        {
          if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
          {
            static Scalar dummy;
            dummy = 0;
            return dummy;
          }
          else
            return Scalar(0);
        }
        else
          return m_argImpl.coeff(row, col);
      }
      else if constexpr(triangle_type == OpenKalman::TriangleType::lower)
      {
        if (row < col)
        {
          if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
          {
            static Scalar dummy;
            dummy = 0;
            return dummy;
          }
          else
            return Scalar(0);
        }
        else
          return m_argImpl.coeff(row, col);
      }
      else
      {
        return m_argImpl.coeff(row, col);
      }
    }

    CoeffReturnType coeff(Index i) const
    {
      return m_argImpl.coeff(i);
    }

  protected:
    NestedEvaluator m_argImpl;
  };


  template<typename ArgType>
  struct evaluator<OpenKalman::EigenDiagonal<ArgType>>
    : evaluator_base<OpenKalman::EigenDiagonal<ArgType>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = OpenKalman::EigenDiagonal<ArgType>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(base_matrix(m_arg)) {}

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
          static Scalar dummy;
          dummy = 0;
          return dummy;
        }
        else
          return Scalar(0);
      }
    }

  protected:
    NestedEvaluator m_argImpl;
  };


  template<typename Nested>
  struct evaluator<OpenKalman::EigenZero<Nested>> : evaluator<typename Nested::ConstantReturnType>
  {
    using XprType = OpenKalman::EigenZero<Nested>;
    using Base = evaluator<typename std::decay_t<Nested>::ConstantReturnType>;
    explicit evaluator(const XprType& m_arg) : Base {m_arg} {}
  };


  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::Covariance<Coefficients, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::convert_base_matrix(
      std::declval<const OpenKalman::Covariance<Coefficients, ArgType>&>()))>>
  {
    using XprType = OpenKalman::Covariance<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::convert_base_matrix(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::is_self_adjoint_v<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::convert_base_matrix(m_arg)) {}
  };


  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::SquareRootCovariance<Coefficients, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::convert_base_matrix(
      std::declval<const OpenKalman::SquareRootCovariance<Coefficients, ArgType>&>()))>>
  {
    using XprType = OpenKalman::SquareRootCovariance<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::convert_base_matrix(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::is_triangular_v<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::convert_base_matrix(m_arg)) {}
  };


  /////////////////////////
  //// ToEuclideanExpr ////
  /////////////////////////

  namespace detail
  {
    // Base class for FromEuclideanExpr evaluators.
    template<typename Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base : evaluator_base<XprType>
    {
      using Scalar = typename Nested::Scalar;

      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : m_argImpl {t} {}

      Scalar& coeffRef(Index row, Index col)
      {
        static_assert(Coefficients::axes_only, "Direct access not available for means containing angles.");
        return m_argImpl.coeffRef(row, col);
      }

      Scalar& coeffRef(Index row)
      {
        static_assert(Coefficients::axes_only, "Direct access not available for means containing angles.");
        return m_argImpl.coeffRef(row);
      }

      template<int LoadMode, typename PacketType>
      PacketType packet(Index row, Index col) const
      {
        static_assert(Coefficients::axes_only, "Packet access not available for means containing angles.");
        return m_argImpl.template packet<LoadMode, PacketType>(col, row);
      }

      template<int LoadMode, typename PacketType>
      PacketType packet(Index index) const
      {
        static_assert(Coefficients::axes_only, "Packet access not available for means containing angles.");
        return m_argImpl.template packet<LoadMode, PacketType>(index);
      }

      template<int StoreMode, typename PacketType>
      void writePacket(Index row, Index col, const PacketType& x)
      {
        static_assert(Coefficients::axes_only, "Packet access not available for means containing angles.");
        m_argImpl.template writePacket<StoreMode, PacketType>(col, row, x);
      }

      template<int StoreMode, typename PacketType>
      void writePacket(Index index, const PacketType& x)
      {
        static_assert(Coefficients::axes_only, "Packet access not available for means containing angles.");
        m_argImpl.template writePacket<StoreMode, PacketType>(index, x);
      }

    protected:
      NestedEvaluator m_argImpl;
    };

  }

  /// Evaluator for ToEuclideanExpr
  /// @tparam Coefficients Coefficient types
  /// @tparam Nested Type of the nested expression
  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::ToEuclideanExpr<Coefficients, ArgType>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator>;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        Coefficients::axes_only ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_sin_op<Scalar>>::Cost +
        Eigen::internal::functor_traits<Eigen::internal::scalar_cos_op<Scalar>>::Cost),
      Flags = Coefficients::axes_only ?
        NestedEvaluator::Flags :
        ColMajor | (count == 1 ? LinearAccessBit : 0),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& t) : Base {base_matrix(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this](const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::to_Euclidean<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row);
      }
      else
      {
        const auto get_coeff = [this](const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::to_Euclidean<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }
  };


  ///////////////////////////
  //// FromEuclideanExpr ////
  ///////////////////////////

  /// General evaluator for FromEuclideanExpr
  /// @tparam Coefficients Coefficient types
  /// @tparam Nested Type of the nested expression
  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coefficients, OpenKalman::FromEuclideanExpr<Coefficients, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<Coefficients, ArgType>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator>;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        Coefficients::axes_only ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_atan_op<Scalar>>::Cost),
      Flags = Coefficients::axes_only ?
        NestedEvaluator::Flags :
        ColMajor | (count == 1 ? LinearAccessBit : 0),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& t) : Base {base_matrix(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this](const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::from_Euclidean<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row);
      }
      else
      {
        const auto get_coeff = [this](const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::from_Euclidean<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }
  };


  /// Specialized evaluator for FromEuclideanExpr that has a nested ToEuclideanExpr.
  /// This amounts to wrapping angles.
  /// @tparam Coefficients Coefficient types
  /// @tparam Nested Type of the nested expression
  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>>
    : detail::Evaluator_EuclideanExpr_Base<Coefficients, OpenKalman::FromEuclideanExpr<Coefficients,
    OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>, std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator>;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = (count == 1 ? LinearAccessBit : 0) | (Coefficients::axes_only ? NestedEvaluator::Flags : (unsigned int) ColMajor),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& t) : Base {base_matrix(base_matrix(t))} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this](const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::wrap<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coefficients::axes_only)
      {
        return this->m_argImpl.coeff(row);
      }
      else
      {
        const auto get_coeff = [this](const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::wrap<Coefficients, Scalar>((std::size_t) row, get_coeff);
      }
    }
  };

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, const OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>> {};

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>> {};

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>&&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>> {};

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, const OpenKalman::ToEuclideanExpr<Coefficients, ArgType>&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>> {};

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coefficients, const OpenKalman::ToEuclideanExpr<Coefficients, ArgType>&&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coefficients, OpenKalman::ToEuclideanExpr<Coefficients, ArgType>>> {};


}

#endif //OPENKALMAN_EIGENEVALUATORS_H
