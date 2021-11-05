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

  // ------------------ //
  //  OpenKalman::Mean  //
  // ------------------ //

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::Mean<Coefficients, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Mean<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    enum
    {
      Flags = (Coefficients::axes_only ? Base::Flags : Base::Flags & ~LvalueBit),
    };
    explicit evaluator(const XprType& m) : Base(m.nested_matrix()) {}
  };


  // -------------------- //
  //  OpenKalman::Matrix  //
  // -------------------- //

  template<typename RowCoeffs, typename ColCoeffs, typename ArgType>
  struct evaluator<OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& m) : Base {m.nested_matrix()} {}
  };


  // --------------------------- //
  //  OpenKalman::EuclideanMean  //
  // --------------------------- //

  template<typename Coeffs, typename ArgType>
  struct evaluator<OpenKalman::EuclideanMean<Coeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = OpenKalman::EuclideanMean<Coeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& expression) : Base {expression.nested_matrix()} {}
  };


  // ------------------------------------ //
  //  OpenKalman::Eigen3::ConstantMatrix  //
  // ------------------------------------ //

  template<typename Scalar_, auto constant, std::size_t rows, std::size_t cols>
  struct evaluator<Eigen3::ConstantMatrix<Scalar_, constant, rows, cols>>
    : evaluator_base<Eigen3::ConstantMatrix<Scalar_, constant, rows, cols>>
  {
    using Scalar = Scalar_;
    using XprType = Eigen3::ConstantMatrix<Scalar, constant, rows, cols>;

    enum {
      CoeffReadCost = 0,
      Flags = NoPreferredStorageOrderBit | LinearAccessBit |
        (traits<Eigen3::eigen_matrix_t<Scalar, rows, cols>>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      Alignment = AlignedMax
    };

    explicit evaluator(const XprType&) {}

    constexpr Scalar coeff(Index row, Index col) const
    {
      return constant;
    }

    constexpr Scalar coeff(Index row) const
    {
      return constant;
    }

    template<int LoadMode, typename PacketType>
    PacketType packet(Index row, Index col) const
    {
      return internal::pset1<PacketType>(constant);
    }

    template<int LoadMode, typename PacketType>
    PacketType packet(Index row) const
    {
      return internal::pset1<PacketType>(constant);
    }

  };


  // -------------------------------- //
  //  OpenKalman::Eigen3::ZeroMatrix  //
  // -------------------------------- //

  template<typename Scalar_, std::size_t rows, std::size_t cols>
  struct evaluator<Eigen3::ZeroMatrix<Scalar_, rows, cols>>
    : evaluator_base<Eigen3::ZeroMatrix<Scalar_, rows, cols>>
  {
    using Scalar = Scalar_;
    using XprType = Eigen3::ZeroMatrix<Scalar, rows, cols>;

    enum {
      CoeffReadCost = 0,
      Flags = NoPreferredStorageOrderBit | EvalBeforeNestingBit | LinearAccessBit |
        (traits<Eigen3::eigen_matrix_t<Scalar, rows, cols>>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      Alignment = AlignedMax
    };

    explicit evaluator(const XprType&) {}

    constexpr Scalar coeff(Index row, Index col) const
    {
      return 0;
    }

    constexpr Scalar coeff(Index row) const
    {
      return 0;
    }

    template<int LoadMode, typename PacketType>
    PacketType packet(Index row, Index col) const
    {
      return internal::pset1<PacketType>(0);
    }

    template<int LoadMode, typename PacketType>
    PacketType packet(Index row) const
    {
      return internal::pset1<PacketType>(0);
    }

  };


  // --------------------------------------- //
  //  OpenKalman::Eigen3::SelfAdjointMatrix  //
  // --------------------------------------- //

  template<typename ArgType, OpenKalman::TriangleType storage_triangle>
  struct evaluator<Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>>
    : evaluator_base<Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;

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
            static constexpr Scalar dummy = 0;
            return dummy;
          }
          else return Scalar {0};
        }
      }
      else if constexpr (storage_triangle == TriangleType::upper)
      {
        if (row > col)
        {
          if constexpr (OpenKalman::complex_number<Scalar>) return std::conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }
      else if constexpr (storage_triangle == TriangleType::lower)
      {
        if (row < col)
        {
          if constexpr (OpenKalman::complex_number<Scalar>) return std::conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }

      if constexpr (OpenKalman::complex_number<Scalar>)
      {
        if (row == col)
        {
          return std::real(m_argImpl.coeff(row, col));
        }
      }

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


  // -------------------------------------- //
  //  OpenKalman::Eigen3::TriangularMatrix  //
  // -------------------------------------- //

  template<typename ArgType, OpenKalman::TriangleType triangle_type>
  struct evaluator<Eigen3::TriangularMatrix<ArgType, triangle_type>>
    : evaluator_base<Eigen3::TriangularMatrix<ArgType, triangle_type>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = Eigen3::TriangularMatrix<ArgType, triangle_type>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
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
        static constexpr Scalar dummy = 0;
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


  // ------------------------------------ //
  //  OpenKalman::Eigen3::DiagonalMatrix  //
  // ------------------------------------ //

  template<typename ArgType>
  struct evaluator<Eigen3::DiagonalMatrix<ArgType>>
    : evaluator_base<Eigen3::DiagonalMatrix<ArgType>>
  {
    using Scalar = typename std::decay_t<ArgType>::Scalar;
    using XprType = Eigen3::DiagonalMatrix<ArgType>;
    using CoeffReturnType = typename XprType::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
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


  // ------------------------ //
  //  OpenKalman::Covariance  //
  // ------------------------ //

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::Covariance<Coefficients, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::Covariance<Coefficients, ArgType>&>()))>>
  {
    using XprType = OpenKalman::Covariance<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::self_adjoint_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::to_covariance_nestable(m_arg)) {}
  };


  // ---------------------------------- //
  //  OpenKalman::SquareRootCovariance  //
  // ---------------------------------- //

  template<typename Coefficients, typename ArgType>
  struct evaluator<OpenKalman::SquareRootCovariance<Coefficients, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::SquareRootCovariance<Coefficients, ArgType>&>()))>>
  {
    using XprType = OpenKalman::SquareRootCovariance<Coefficients, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::triangular_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::to_covariance_nestable(m_arg)) {}
  };


  // --------------------------------------- //
  //  Base classes for euclidean evaluators  //
  // --------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<OpenKalman::coefficients Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
#else
    template<typename Coefficients, typename XprType, typename Nested, typename NestedEvaluator, typename = void>
#endif
    struct Evaluator_EuclideanExpr_Base;


#ifdef __cpp_concepts
    template<OpenKalman::coefficients Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
      requires (not Coefficients::axes_only)
    struct Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator>
#else
    template<typename Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator,
      std::enable_if_t<OpenKalman::coefficients<Coefficients> and not Coefficients::axes_only>>
#endif
      : evaluator_base<XprType>
    {
      using CoeffReturnType = typename Nested::Scalar;

      enum
      {
        Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
          (~(OpenKalman::MatrixTraits<Nested>::columns == 1 ? 0 : LinearAccessBit)),
        Alignment = NestedEvaluator::Alignment
      };

      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : m_argImpl {t} {}

    protected:

      NestedEvaluator m_argImpl;
    };


#ifdef __cpp_concepts
    template<OpenKalman::coefficients Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
      requires Coefficients::axes_only
    struct Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator>
#else
    template<typename Coefficients, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<Coefficients, XprType, Nested, NestedEvaluator,
      std::enable_if_t<OpenKalman::coefficients<Coefficients> and Coefficients::axes_only>>
#endif
      : NestedEvaluator
    {
      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : NestedEvaluator {t} {}
    };

  }


  // ------------------------- //
  //  Eigen3::ToEuclideanExpr  //
  // ------------------------- //

  /**
   * \internal
   * \brief Evaluator for ToEuclideanExpr
   * \tparam Coeffs Coefficient types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<Eigen3::ToEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = Eigen3::ToEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        Coeffs::axes_only ? 0 :
        (int) Eigen::internal::functor_traits<Eigen::internal::scalar_sin_op<Scalar>>::Cost +
          (int) Eigen::internal::functor_traits<Eigen::internal::scalar_cos_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_matrix()} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this] (const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::internal::to_euclidean_coeff<Coeffs>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto get_coeff = [this] (const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::internal::to_euclidean_coeff<Coeffs>((std::size_t) row, get_coeff);
      }
    }

  };


  // --------------------------------------- //
  //  OpenKalman::Eigen3::FromEuclideanExpr  //
  // --------------------------------------- //

  /**
   * \internal
   * \brief General evaluator for FromEuclideanExpr
   * \tparam Coeffs Coefficient types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<Eigen3::FromEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, Eigen3::FromEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = Eigen3::FromEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        Coeffs::axes_only ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_atan_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_matrix()} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this] (const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::internal::from_euclidean_coeff<Coeffs>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto get_coeff = [this] (const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::internal::from_euclidean_coeff<Coeffs>((std::size_t) row, get_coeff);
      }
    }
  };


  /**
   * \internal
   * \brief Specialized evaluator for FromEuclideanExpr that has a nested ToEuclideanExpr.
   * \details This amounts to wrapping angles.
   * \tparam Coeffs Coefficient types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<
    Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
      : detail::Evaluator_EuclideanExpr_Base<Coeffs,
        Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>,
        std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename XprType::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost
    };

    template<typename Arg>
    explicit evaluator(const Arg& t) : Base {t.nested_matrix().nested_matrix()} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto get_coeff = [col, this] (const Index i) { return this->m_argImpl.coeff(i, col); };
        return OpenKalman::internal::wrap_get<Coeffs>((std::size_t) row, get_coeff);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (Coeffs::axes_only)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto get_coeff = [this] (const Index i) { return this->m_argImpl.coeff(i); };
        return OpenKalman::internal::wrap_get<Coeffs>((std::size_t) row, get_coeff);
      }
    }
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    Eigen3::FromEuclideanExpr<Coeffs, const Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
    : evaluator<Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<Eigen3::FromEuclideanExpr<Coeffs,
      Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<Eigen3::FromEuclideanExpr<Coeffs,
      Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    Eigen3::FromEuclideanExpr<Coeffs, const Eigen3::ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<Eigen3::FromEuclideanExpr<Coeffs, Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<Eigen3::FromEuclideanExpr<Coeffs,
      Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


} // namespace Eigen::internal

#endif //OPENKALMAN_EIGEN3_NATIVE_EVALUATORS_HPP
