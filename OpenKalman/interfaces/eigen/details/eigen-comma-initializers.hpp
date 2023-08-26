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
 * \brief Definitions for extensions to Eigen::CommaInitializer
 */

#ifndef OPENKALMAN_EIGEN_COMMA_INITIALIZERS_HPP
#define OPENKALMAN_EIGEN_COMMA_INITIALIZERS_HPP

namespace Eigen
{
  /**
   * \brief Alternative version of Eigen::CommaInitializer for Mean.
   */
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer : CommaInitializer<XprType>
  {
    using Base = CommaInitializer<XprType>;
    using Scalar = OpenKalman::scalar_type_of_t<XprType>;
    using TypedIndex = OpenKalman::row_index_descriptor_of_t<Derived>;
    using Base::Base;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    MeanCommaInitializer(XprType& xpr, const S& s) : Base {xpr, static_cast<const Scalar&>(s)} {}

    template<typename OtherDerived>
    MeanCommaInitializer(XprType& xpr, const DenseBase<OtherDerived>& other)
      : Base {xpr, other} {}

    ~MeanCommaInitializer()
    {
      this->m_xpr = OpenKalman::wrap_angles<TypedIndex>(Base::finished());
    }

    auto& finished()
    {
      this->m_xpr = OpenKalman::wrap_angles<TypedIndex>(Base::finished());
      return this->m_xpr;
    }
  };


  /**
   * \brief Version of Eigen::CommaInitializer for diagonal versions of SelfAdjointMatrix and TriangularMatrix.
   */
  template<typename XprType>
  struct DiagonalCommaInitializer
  {
    using Scalar = OpenKalman::scalar_type_of_t<XprType>;
    static constexpr auto dim = OpenKalman::row_dimension_of_v<XprType>;
    using NestedMatrix = OpenKalman::untyped_dense_writable_matrix_t<OpenKalman::pattern_matrix_of_t<XprType>, Scalar, dim, 1>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    XprType& diag;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    DiagonalCommaInitializer(XprType& xpr, const S& s)
      : matrix {}, comma_initializer {matrix, static_cast<const Scalar&>(s)}, diag {xpr} {}

    template<typename OtherDerived>
    DiagonalCommaInitializer(XprType& xpr, const DenseBase<OtherDerived>& other)
      : matrix {}, comma_initializer {matrix, other}, diag {xpr} {}

    DiagonalCommaInitializer(const DiagonalCommaInitializer& o)
      : matrix {o.matrix}, comma_initializer {o.comma_initializer}, diag {o.diag} {}

    DiagonalCommaInitializer(DiagonalCommaInitializer&& o)
      : matrix {std::move(o.matrix)},
        comma_initializer {std::move(o.comma_initializer)}, diag {std::move(o.diag)} {}

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~DiagonalCommaInitializer()
    {
      diag = OpenKalman::DiagonalMatrix<NestedMatrix>(comma_initializer.finished());
    }

    auto& finished()
    {
      diag = OpenKalman::DiagonalMatrix<NestedMatrix>(comma_initializer.finished());
      return diag;
    }
  };


  /**
   * \brief Alternative version of Eigen::CommaInitializer for Covariance and SquareRootCovariance.
   */
  template<typename CovarianceType>
  struct CovarianceCommaInitializer
  {
    using Scalar = OpenKalman::scalar_type_of_t<CovarianceType>;
    using CovNest = OpenKalman::nested_matrix_of_t<CovarianceType>;
    using NestedMatrix = std::conditional_t<OpenKalman::diagonal_matrix<CovNest>,
      OpenKalman::untyped_dense_writable_matrix_t<CovNest, Scalar, OpenKalman::row_dimension_of_v<CovNest>, 1>,
      OpenKalman::dense_writable_matrix_t<CovNest, Scalar>>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    CovarianceType& cov;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(CovarianceType& xpr, const S& s)
      : matrix {}, comma_initializer {matrix, static_cast<const Scalar&>(s)}, cov {xpr} {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(CovarianceType& xpr, const DenseBase<OtherDerived>& other)
      : matrix {}, comma_initializer {matrix, other}, cov {xpr} {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix {o.matrix}, comma_initializer {o.comma_initializer}, cov {o.cov} {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix {std::move(o.matrix)},
        comma_initializer {std::move(o.comma_initializer)}, cov {std::move(o.cov)} {}

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~CovarianceCommaInitializer()
    {
      using namespace OpenKalman;

      if constexpr (diagonal_matrix<CovNest>)
      {
        cov = CovarianceType {std::move(comma_initializer).finished()};
      }
      else if constexpr (triangular_covariance<CovarianceType>)
      {
        using T = std::conditional_t<triangular_matrix<CovNest>, CovNest, decltype(Cholesky_factor(std::declval<CovNest>()))>;
        auto b = OpenKalman::internal::to_covariance_nestable<T>(std::move(comma_initializer).finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      else
      {
        using SA = std::conditional_t<hermitian_matrix<CovNest>, CovNest, decltype(Cholesky_square(std::declval<CovNest>()))>;
        auto b = OpenKalman::internal::to_covariance_nestable<SA>(std::move(comma_initializer).finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
    }

    auto& finished()
    {
      using namespace OpenKalman;

      if constexpr (diagonal_matrix<CovNest>)
      {
        cov = CovarianceType {comma_initializer.finished()};
      }
      else if constexpr (triangular_covariance<CovarianceType>)
      {
        using T = std::conditional_t<triangular_matrix<CovNest>, CovNest, decltype(Cholesky_factor(std::declval<CovNest>()))>;
        auto b = OpenKalman::internal::to_covariance_nestable<T>(comma_initializer.finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      else
      {
        using SA = std::conditional_t<hermitian_matrix<CovNest>, CovNest, decltype(Cholesky_square(std::declval<CovNest>()))>;
        auto b = OpenKalman::internal::to_covariance_nestable<SA>(comma_initializer.finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      return cov;
    }
  };

} // namespace Eigen


#endif //OPENKALMAN_EIGEN_COMMA_INITIALIZERS_HPP
