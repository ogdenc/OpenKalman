/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "VectorSpaceAdapter.hpp"

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
    template<OpenKalman::vector_space_descriptor V0, typename XprType, typename Nested, typename NestedEvaluator>
#else
    template<typename V0, typename XprType, typename Nested, typename NestedEvaluator, typename = void>
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
    template<OpenKalman::euclidean_vector_space_descriptor V0, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<V0, XprType, Nested, NestedEvaluator>
#else
    template<typename V0, typename XprType, typename Nested, typename NestedEvaluator>
    struct Evaluator_EuclideanExpr_Base<V0, XprType, Nested, NestedEvaluator,
      std::enable_if_t<OpenKalman::euclidean_vector_space_descriptor<V0>>>
#endif
      : NestedEvaluator
    {
      explicit Evaluator_EuclideanExpr_Base(const Nested& t) : NestedEvaluator {t} {}
    };

  } // namespace detail


  // ----------------- //
  //  ToEuclideanExpr  //
  // ----------------- //

  /**
   * \internal
   * \brief Evaluator for ToEuclideanExpr
   */
  template<typename ArgType>
  struct evaluator<OpenKalman::ToEuclideanExpr<ArgType>>
    : detail::Evaluator_EuclideanExpr_Base<OpenKalman::vector_space_descriptor_of<ArgType, 0>, OpenKalman::ToEuclideanExpr<ArgType>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
	using V0 = OpenKalman::vector_space_descriptor_of<ArgType, 0>; 
    using XprType = OpenKalman::ToEuclideanExpr<ArgType>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<V0, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    V0 i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        OpenKalman::euclidean_vector_space_descriptor<V0> ? 0 :
        (int) Eigen::internal::functor_traits<Eigen::internal::scalar_sin_op<Scalar>>::Cost +
          (int) Eigen::internal::functor_traits<Eigen::internal::scalar_cos_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return descriptor::to_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return descriptor::to_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }

  };


  // ------------------- //
  //  FromEuclideanExpr  //
  // ------------------- //

  /**
   * \internal
   * \brief General evaluator for FromEuclideanExpr
   */
  template<typename ArgType, typename V0>
  struct evaluator<OpenKalman::FromEuclideanExpr<ArgType, V0>>
    : detail::Evaluator_EuclideanExpr_Base<V0, OpenKalman::FromEuclideanExpr<ArgType, V0>,
    std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<ArgType, V0>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<V0, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    V0 i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost + (
        OpenKalman::euclidean_vector_space_descriptor<V0> ? 0 :
        Eigen::internal::functor_traits<Eigen::internal::scalar_atan_op<Scalar>>::Cost)
    };

    explicit evaluator(const XprType& t) : Base {t.nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return i_vector_space_descriptor.from_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return descriptor::from_euclidean_element(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }
  };


  /**
   * \internal
   * \brief Specialized evaluator for FromEuclideanExpr that has a nested ToEuclideanExpr.
   * \details This amounts to wrapping angles.
   */
  template<typename ArgType, typename V0>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>
      : detail::Evaluator_EuclideanExpr_Base<V0,
        OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>,
        std::decay_t<ArgType>, evaluator<std::decay_t<ArgType>>>
  {
    using XprType = OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>;
    using Nested = std::decay_t<ArgType>;
    using NestedEvaluator = evaluator<Nested>;
    using Base = detail::Evaluator_EuclideanExpr_Base<V0, XprType, Nested, NestedEvaluator>;
    using typename Base::CoeffReturnType;
    using Scalar = typename traits<Nested>::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;

    V0 i_vector_space_descriptor;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost
    };

    template<typename Arg>
    explicit evaluator(const Arg& t) : Base {t.nested_object().nested_object()}, i_vector_space_descriptor {OpenKalman::get_vector_space_descriptor<0>(t)} {}

    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row, col);
      }
      else
      {
        const auto g = [col, this] (std::size_t i) { return this->m_argImpl.coeff((Index) i, col); };
        return descriptor::get_wrapped_component(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }

    CoeffReturnType coeff(Index row) const
    {
      if constexpr (OpenKalman::euclidean_vector_space_descriptor<V0>)
      {
        return Base::coeff(row);
      }
      else
      {
        const auto g = [this] (std::size_t i) { return this->m_argImpl.coeff((Index) i); };
        return descriptor::get_wrapped_component(i_vector_space_descriptor, g, (std::size_t) row);
      }
    }
  };


  template<typename ArgType, typename V0>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<const OpenKalman::ToEuclideanExpr<ArgType>, V0>>
    : evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>;
    using Base::Base;
  };


  template<typename ArgType, typename V0>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>&, V0>>
    : evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>;
    using Base::Base;
  };


  template<typename ArgType, typename V0>
  struct evaluator<
    OpenKalman::FromEuclideanExpr<const OpenKalman::ToEuclideanExpr<ArgType>&, V0>>
    : evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>
  {
    using Base = evaluator<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<ArgType>, V0>>;
    using Base::Base;
  };


  // ------ //
  //  Mean  //
  // ------ //

  template<typename StaticDescriptor, typename ArgType>
  struct evaluator<OpenKalman::Mean<StaticDescriptor, ArgType>> : evaluator<std::decay_t<ArgType>>
  {
    using XprType = OpenKalman::Mean<StaticDescriptor, ArgType>;
    using Base = evaluator<std::decay_t<ArgType>>;
    enum
    {
      Flags = (OpenKalman::euclidean_vector_space_descriptor<StaticDescriptor> ? Base::Flags : Base::Flags & ~LvalueBit),
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

  template<typename StaticDescriptor, typename ArgType>
  struct evaluator<OpenKalman::Covariance<StaticDescriptor, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::Covariance<StaticDescriptor, ArgType>&>()))>>
  {
    using XprType = OpenKalman::Covariance<StaticDescriptor, ArgType>;
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

  template<typename StaticDescriptor, typename ArgType>
  struct evaluator<OpenKalman::SquareRootCovariance<StaticDescriptor, ArgType>>
    : evaluator<std::decay_t<decltype(OpenKalman::internal::to_covariance_nestable(
      std::declval<const OpenKalman::SquareRootCovariance<StaticDescriptor, ArgType>&>()))>>
  {
    using XprType = OpenKalman::SquareRootCovariance<StaticDescriptor, ArgType>;
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
