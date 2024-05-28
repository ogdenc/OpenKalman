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

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HPP

#include <complex>

namespace Eigen::internal
{
  // ------------------ //
  //  FixedSizeAdapter  //
  // ------------------ //

  template<typename NestedMatrix, typename...Vs>
  struct evaluator<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Vs...>>
    : evaluator<std::decay_t<NestedMatrix>>
  {
    using XprType = OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Vs...>;
    using Base = evaluator<std::decay_t<NestedMatrix>>;
    explicit evaluator(const XprType& arg) : Base {OpenKalman::nested_object(arg)} {}
  };


  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct evaluator<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
    : evaluator_base<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
  {
    using XprType = OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>;
    using M = OpenKalman::Eigen3::eigen_matrix_t<Scalar, OpenKalman::index_dimension_of_v<PatternMatrix, 0>,
      OpenKalman::index_dimension_of_v<PatternMatrix, 1>>;

    enum {
      CoeffReadCost = 0,
      Flags = NoPreferredStorageOrderBit | LinearAccessBit | (traits<M>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      Alignment = AlignedMax
    };

    explicit constexpr evaluator(const XprType& t) : m_xpr{t} {}

    constexpr Scalar coeff(Index row, Index col) const
    {
      return m_xpr.get_scalar_constant();
    }

    constexpr Scalar coeff(Index row) const
    {
      return m_xpr.get_scalar_constant();
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      return internal::pset1<PacketType>(m_xpr.get_scalar_constant());
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row) const
    {
      return internal::pset1<PacketType>(m_xpr.get_scalar_constant());
    }

  protected:

    const XprType& m_xpr;
  };


  // ------------------- //
  //  SelfAdjointMatrix  //
  // ------------------- //

  template<typename ArgType, OpenKalman::HermitianAdapterType storage_triangle>
  struct evaluator<OpenKalman::SelfAdjointMatrix<ArgType, storage_triangle>>
    : evaluator_base<OpenKalman::SelfAdjointMatrix<ArgType, storage_triangle>>
  {
    using XprType = OpenKalman::SelfAdjointMatrix<ArgType, storage_triangle>;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };


    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_object()) {}


    auto& coeffRef(Index row, Index col)
    {
      static_assert(not OpenKalman::complex_number<Scalar>,
        "Reference to element is not available for a complex SelfAdjointMatrix");

      if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::upper)
      {
        if (row > col)
          return m_argImpl.coeffRef(col, row);
      }
      else if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::lower)
      {
        if (row < col)
          return m_argImpl.coeffRef(col, row);
      }

      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one SelfAdjointMatrix");

      static_assert(not OpenKalman::complex_number<Scalar>,
        "Reference to element is not available for a complex SelfAdjointMatrix");

      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::upper)
      {
        if (row > col)
        {
          using std::conj;
          if constexpr (OpenKalman::complex_number<Scalar>) return conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }
      else if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::lower)
      {
        if (row < col)
        {
          using std::conj;
          if constexpr (OpenKalman::complex_number<Scalar>) return conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }

      if (row == col)
      {
        if constexpr (OpenKalman::complex_number<Scalar> and not std::is_lvalue_reference_v<CoeffReturnType>)
        {
          using std::real;
          return real(m_argImpl.coeff(row, col));
        }
      }

      return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access is only available for one-by-one SelfAdjointMatrix");

      return m_argImpl.coeff(i);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Packet access is only available for one-by-one SelfAdjointMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear packet access is only available for one-by-one SelfAdjointMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(index);
    }

  protected:

    NestedEvaluator m_argImpl;
  };


  // ------------------ //
  //  TriangularMatrix  //
  // ------------------ //

  template<typename ArgType, OpenKalman::TriangleType triangle_type>
  struct evaluator<OpenKalman::TriangularMatrix<ArgType, triangle_type>>
    : evaluator_base<OpenKalman::TriangularMatrix<ArgType, triangle_type>>
  {
    using XprType = OpenKalman::TriangularMatrix<ArgType, triangle_type>;
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };

    //static constexpr bool is_row_major = static_cast<bool>(NestedEvaluator::Flags & RowMajorBit);


    explicit evaluator(const XprType& m_arg) : m_argImpl {m_arg.nested_object()} {}


    auto& coeffRef(Index row, Index col)
    {
      static Scalar dummy;
      if constexpr(triangle_type == OpenKalman::TriangleType::diagonal) {if (row != col) { dummy = 0; return dummy; }}
      else if constexpr(triangle_type == OpenKalman::TriangleType::upper) {if (row > col) { dummy = 0; return dummy; }}
      else if constexpr(triangle_type == OpenKalman::TriangleType::lower) {if (row < col) { dummy = 0; return dummy; }}
      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one TriangularMatrix");

      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
      {
        static std::remove_reference_t<CoeffReturnType> dummy = 0;
        if constexpr(triangle_type == OpenKalman::TriangleType::diagonal) {if (row != col) return dummy;}
        else if constexpr(triangle_type == OpenKalman::TriangleType::upper) {if (row > col) return dummy;}
        else if constexpr(triangle_type == OpenKalman::TriangleType::lower) {if (row < col) return dummy;}
      }
      else
      {
        if constexpr(triangle_type == OpenKalman::TriangleType::diagonal) {if (row != col) return Scalar(0);}
        else if constexpr(triangle_type == OpenKalman::TriangleType::upper) {if (row > col) return Scalar(0);}
        else if constexpr(triangle_type == OpenKalman::TriangleType::lower) {if (row < col) return Scalar(0);}
      }
      return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access is only available for one-by-one TriangularMatrix");

      return m_argImpl.coeff(i);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Packet access is only available for one-by-one TriangularMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear packet access is only available for one-by-one TriangularMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(index);
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
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_object()) {}

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

    auto& coeffRef(Index index)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one DiagonalMatrix");

      return m_argImpl.coeffRef(index);
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

    CoeffReturnType coeff(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access is only available for one-by-one DiagonalMatrix");

      return m_argImpl.coeff(index);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Packet access is only available for one-by-one DiagonalMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear packet access is only available for one-by-one DiagonalMatrix");

      return m_argImpl.template packet<LoadMode, PacketType>(index);
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
    template<OpenKalman::vector_space_descriptor FixedDescriptor, typename XprType, typename Nested, typename NestedEvaluator>
#else
    template<typename FixedDescriptor, typename XprType, typename Nested, typename NestedEvaluator, typename = void>
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
    template<OpenKalman::euclidean_vector_space_descriptor FixedDescriptor, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<FixedDescriptor, XprType, Nested, NestedEvaluator>
#else
    template<typename FixedDescriptor, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<FixedDescriptor, XprType, Nested, NestedEvaluator,
      std::enable_if_t<OpenKalman::euclidean_vector_space_descriptor<FixedDescriptor>>>
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
   * \tparam Coeffs FixedDescriptor types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::ToEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        OpenKalman::euclidean_vector_space_descriptor<Coeffs> ? 0 :
        (int) Eigen::internal::functor_traits<Eigen::internal::scalar_sin_op<Scalar>>::Cost +
          (int) Eigen::internal::functor_traits<Eigen::internal::scalar_cos_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return to_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return to_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }

  };


  // ------------------- //
  //  FromEuclideanExpr  //
  // ------------------- //

  /**
   * \internal
   * \brief General evaluator for FromEuclideanExpr
   * \tparam Coeffs FixedDescriptor types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<OpenKalman::FromEuclideanExpr<Coeffs, ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<Coeffs, OpenKalman::FromEuclideanExpr<Coeffs, ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<Coeffs, ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        OpenKalman::euclidean_vector_space_descriptor<Coeffs> ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_atan_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return i_vector_space_descriptor.from_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return from_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }
  };


  /**
   * \internal
   * \brief Specialized evaluator for FromEuclideanExpr that has a nested ToEuclideanExpr.
   * \details This amounts to wrapping angles.
   * \tparam Coeffs FixedDescriptor types
   * \tparam ArgType Type of the nested expression
   */
  template<typename Coeffs, typename ArgType>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
      : detail::Evaluator_EuclideanExpr_Base<Coeffs,
        OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>,
        std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<Coeffs, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    Coeffs i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost
    };

    template<typename Arg>
    explicit evaluator(const Arg& t) : Base {t.nested_object().nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return get_wrapped_component(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<Coeffs>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return get_wrapped_component(i_vector_space_descriptor, g, (std::size_t) row, 0);
      }
    }
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<Coeffs, const OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<Coeffs,
      OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  template<typename Coeffs, typename ArgType>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<Coeffs, const OpenKalman::ToEuclideanExpr<Coeffs, ArgType>&>>
    : evaluator<OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<Coeffs,
      OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>;
    using Base::Base;
  };


  // ------ //
  //  Mean  //
  // ------ //

  template<typename FixedDescriptor, typename ArgType>
  struct evaluator<OpenKalman::Mean<FixedDescriptor, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Mean<FixedDescriptor, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    enum
    {
      Flags = (OpenKalman::euclidean_vector_space_descriptor<FixedDescriptor> ? Base::Flags : Base::Flags & ~LvalueBit),
    };
    explicit evaluator(const XprType& m) : Base(m.nested_object()) {}
  };


  // -------- //
  //  Matrix  //
  // -------- //

  template<typename RowCoeffs, typename ColCoeffs, typename ArgType>
  struct evaluator<OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Matrix<RowCoeffs, ColCoeffs, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    explicit evaluator(const XprType& m) : Base {m.nested_object()} {}
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
    explicit evaluator(const XprType& expression) : Base {expression.nested_object()} {}
  };


  // ------------ //
  //  Covariance  //
  // ------------ //

  template<typename FixedDescriptor, typename ArgType>
  struct evaluator<OpenKalman::Covariance<FixedDescriptor, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::Covariance<FixedDescriptor, ArgType>&>()))>>
  {
    using XprType = OpenKalman::Covariance<FixedDescriptor, ArgType>;
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

  template<typename FixedDescriptor, typename ArgType>
  struct evaluator<OpenKalman::SquareRootCovariance<FixedDescriptor, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::SquareRootCovariance<FixedDescriptor, ArgType>&>()))>>
  {
    using XprType = OpenKalman::SquareRootCovariance<FixedDescriptor, ArgType>;
    using Base = evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const XprType&>()))>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::triangular_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
    explicit evaluator(const XprType& m_arg) : Base(OpenKalman::internal::to_covariance_nestable(m_arg)) {}
  };


} // namespace Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HPP
