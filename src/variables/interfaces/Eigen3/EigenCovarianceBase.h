/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENCOVARIANCEBASE_H
#define OPENKALMAN_EIGENCOVARIANCEBASE_H

namespace Eigen
{
  template<typename XprApparentType, typename XprActualType>
  struct CovarianceCommaInitializer;
}

namespace OpenKalman::internal
{
  template<typename Derived, typename Nested>
  struct EigenCovarianceBase<Derived, Nested,
    std::enable_if_t<is_self_adjoint_v<Nested> and not is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, Nested> {};

  template<typename Derived, typename ArgType>
  struct EigenCovarianceBase<Derived, ArgType,
    std::enable_if_t<is_triangular_v<ArgType> and not is_diagonal_v<ArgType> and not is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = EigenMatrixBase<Derived, Nested>;
    using Apparent = typename MatrixTraits<Nested>::template SelfAdjointBaseType<>;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = base_matrix(static_cast<Derived&>(*this));
      return Eigen::CovarianceCommaInitializer<Apparent, Nested>(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = base_matrix(static_cast<Derived&>(*this));
      return Eigen::CovarianceCommaInitializer<Apparent, Nested>(xpr, other);
    }
  };

  template<typename Derived, typename ArgType>
  struct EigenCovarianceBase<Derived, ArgType,
    std::enable_if_t<is_self_adjoint_v<ArgType> and not is_diagonal_v<ArgType> and is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = EigenMatrixBase<Derived, Nested>;
    using Apparent = typename MatrixTraits<Nested>::template TriangularBaseType<>;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = base_matrix(static_cast<Derived&>(*this));
      return Eigen::CovarianceCommaInitializer<Apparent, Nested>(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = base_matrix(static_cast<Derived&>(*this));
      return Eigen::CovarianceCommaInitializer<Apparent, Nested>(xpr, other);
    }
  };

  template<typename Derived, typename Nested>
  struct EigenCovarianceBase<Derived, Nested,
    std::enable_if_t<is_triangular_v<Nested> and is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, Nested> {};

}


namespace Eigen
{
  template<typename XprApparentType, typename XprActualType>
  struct CovarianceCommaInitializer
  {
    using Scalar = typename XprApparentType::Scalar;
    using BaseMatrix = typename OpenKalman::MatrixTraits<typename OpenKalman::MatrixTraits<XprApparentType>::BaseMatrix>::template StrictMatrix<>;
    using Nested = CommaInitializer<BaseMatrix>;

    BaseMatrix matrix;
    Nested comma_initializer;
    XprActualType& xpr_actual; // target expression after conversion

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(XprActualType& xpr, const S& s)
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), xpr_actual(xpr) {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(XprActualType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), xpr_actual(xpr) {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), xpr_actual(o.xpr_actual) {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix(std::move(o.matrix)), comma_initializer(std::move(o.comma_initializer)), xpr_actual(std::move(o.xpr_actual)) {}

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
      finished();
    }

    auto& finished()
    {
      return xpr_actual = XprApparentType(comma_initializer.finished());
    }
  };

} // end namespace Eigen


#endif //OPENKALMAN_EIGENCOVARIANCEBASE_H
