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

#include "LibraryWrapper.hpp"
#include "FixedSizeAdapter.hpp"

#include "ConstantAdapter.hpp"
#include "HermitianAdapter.hpp"
#include "TriangularAdapter.hpp"
#include "DiagonalAdapter.hpp"

namespace Eigen::internal
{
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
