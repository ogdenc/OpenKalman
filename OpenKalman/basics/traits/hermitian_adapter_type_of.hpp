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
 * \brief Definition for \ref hermitian_adapter_type_of.
 */

#ifndef OPENKALMAN_HERMITIAN_ADAPTER_TYPE_OF_HPP
#define OPENKALMAN_HERMITIAN_ADAPTER_TYPE_OF_HPP


namespace OpenKalman
{
  /**
   * \brief The TriangleType associated with the storage triangle of one or more matrices.
   * \details If there is no common triangle type, the result is TriangleType::any.
   * If the matrices have a dynamic shape, the result assumes the matrices are square.
   */
  template<typename T, typename...Ts>
  struct hermitian_adapter_type_of : std::integral_constant<HermitianAdapterType,
    (hermitian_adapter<T, HermitianAdapterType::lower> and ... and hermitian_adapter<Ts, HermitianAdapterType::lower>) ? HermitianAdapterType::lower :
    (hermitian_adapter<T, HermitianAdapterType::upper> and ... and hermitian_adapter<Ts, HermitianAdapterType::upper>) ? HermitianAdapterType::upper :
    HermitianAdapterType::any> {};


  /**
   * \brief The TriangleType associated with the storage triangle of a \ref hermitian_matrix.
   * \details Possible values are \ref HermitianAdapterType::lower "lower", \ref HermitianAdapterType::upper "upper", or
   * \ref HermitianAdapterType::any "any".
   */
  template<typename T, typename...Ts>
  constexpr auto hermitian_adapter_type_of_v = hermitian_adapter_type_of<T, Ts...>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_HERMITIAN_ADAPTER_TYPE_OF_HPP
